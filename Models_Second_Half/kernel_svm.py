"""
Kernel SVM Classifier training pipeline for COPD risk prediction.
Uses RBF (Radial Basis Function) kernel with validation.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_data(data_dir: str = None):
    """Load *simple* preprocessed feature matrices and targets."""
    print("\n" + "=" * 80)
    print("LOADING DATA FOR KERNEL SVM")
    print("=" * 80)

    # Get the script's directory and resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if data_dir is None:
        data_dir = os.path.join(project_root, "chronic-obstructive")
    
    # Use simpler, low-dimensional preprocessing to speed up SVM training
    train_path = os.path.join(data_dir, "train_processed_simple.csv")
    test_path = os.path.join(data_dir, "test_processed_simple.csv")
    ids_path = os.path.join(data_dir, "ids_simple.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Simple preprocessing files not found. Run data_preprocessing_simple.py first."
        )

    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    # Target label: prefer column embedded in the simple train file, fallback to train_target.csv
    if "has_copd_risk" in X_train.columns:
        y_train = X_train.pop("has_copd_risk")
    else:
        y_train = pd.read_csv(os.path.join(data_dir, "train_target.csv"))["has_copd_risk"]

    # Test patient IDs from ids_simple if available, otherwise from raw test.csv
    if os.path.exists(ids_path):
        test_patient_ids = pd.read_csv(ids_path)["patient_id"]
    else:
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


def build_kernel_svm_pipeline(kernel='linear', C=1.0, gamma='scale'):
    """
    Build Kernel SVM pipeline with feature scaling.
    Uses LinearSVC for linear kernel (much faster), SVC for others.
    
    Parameters:
    -----------
    kernel : str, default='linear'
        Kernel type: 'linear' (fast), 'rbf', 'poly', 'sigmoid'
    C : float, default=1.0
        Regularization parameter
    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    """
    if kernel == 'linear':
        # Use LinearSVC for linear kernel - much faster than SVC
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    LinearSVC(
                        C=C,
                        class_weight="balanced",
                        random_state=42,
                        max_iter=2000,  # Limit iterations for speed
                        dual=False,  # Faster for n_samples > n_features
                    ),
                ),
            ]
        )
    else:
        # Use SVC for non-linear kernels
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        class_weight="balanced",
                        probability=True,  # Enable probability estimates
                        random_state=42,
                        max_iter=1000,  # Limit iterations for speed
                    ),
                ),
            ]
        )


def create_visualizations(
    model: Pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    kernel: str,
    C: float,
    output_dir: str = None,
):
    """
    Create and save visualization plots for the SVM model.
    
    Parameters:
    -----------
    model : Pipeline
        Trained SVM pipeline
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    kernel : str
        Kernel type used
    C : float
        Regularization parameter used
    output_dir : str, optional
        Directory to save plots. If None, uses chronic-obstructive/model_plots
    """
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "chronic-obstructive", "model_plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Get predictions
    val_pred = model.predict(X_val)
    
    # Try to get probabilities (may not be available for LinearSVC)
    val_proba = None
    if hasattr(model.named_steps['svm'], 'predict_proba'):
        try:
            val_proba = model.predict_proba(X_val)[:, 1]
        except:
            pass
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred, zero_division=0)
    recall = recall_score(y_val, val_pred, zero_division=0)
    f1 = f1_score(y_val, val_pred)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Risk', 'COPD Risk'],
                yticklabels=['No Risk', 'COPD Risk'])
    plt.title(f'Kernel SVM ({kernel.upper()}) - Confusion Matrix\nC={C:.2f}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'kernel_svm_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {cm_path}")
    
    # 2. ROC Curve (if probabilities available)
    if val_proba is not None:
        fpr, tpr, _ = roc_curve(y_val, val_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Kernel SVM ({kernel.upper()}) - ROC Curve\nC={C:.2f}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, 'kernel_svm_roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved ROC curve to {roc_path}")
        
        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Kernel SVM ({kernel.upper()}) - Precision-Recall Curve\nC={C:.2f}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join(output_dir, 'kernel_svm_pr_curve.png')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Precision-Recall curve to {pr_path}")
    
    # 4. Metrics Summary Bar Plot
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Kernel SVM ({kernel.upper()}) - Classification Metrics Summary\nC={C:.2f}', 
             fontsize=14, fontweight='bold')
    plt.ylim([0, 1.0])
    
    # Add value labels on bars
    for bar, (metric, value) in zip(bars, metrics.items()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'kernel_svm_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics summary to {summary_path}")
    
    print(f"\nAll visualizations saved to {output_dir}")


def train_kernel_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: str = 'linear',
    C: float = 1.0,
    gamma: str | float = 'scale',
    use_sample: bool = False,
    sample_size: int = 10000,
    auto_tune_C: bool = True,
    create_plots: bool = True,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Train a Kernel SVM classifier with validation.
    Optimized for fast training (minutes instead of hours).
    
    Parameters:
    -----------
    kernel : str
        Kernel type: 'linear' (fastest), 'rbf', 'poly', 'sigmoid'
    C : float
        Regularization parameter
    gamma : str or float
        Kernel coefficient
    use_sample : bool
        If True, use a sample of data for faster training
    sample_size : int
        Size of sample if use_sample=True
    """
    print("\n" + "=" * 80)
    print("TRAINING KERNEL SVM (OPTIMIZED FOR SPEED)")
    print("=" * 80)

    X_np = X_train.to_numpy(dtype=np.float32)
    y_np = y_train.to_numpy(dtype=np.int64)

    # Optionally use a sample for faster training
    if use_sample and len(X_np) > sample_size:
        print(f"\nUsing sample of {sample_size} samples for faster training...")
        from sklearn.utils import resample
        X_sample, y_sample = resample(X_np, y_np, n_samples=sample_size, 
                                      stratify=y_np, random_state=42)
        X_np = X_sample
        y_np = y_sample

    # Split data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=0.2,
        stratify=y_np,
        random_state=42,
    )

    print(f"Training set: {X_tr.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Kernel: {kernel.upper()}")
    print(f"Initial C (regularization): {C}")
    if kernel != 'linear':
        print(f"Gamma: {gamma}")

    # Optionally do a tiny one-pass C search on the validation split (keeps it fast)
    if auto_tune_C and kernel == 'linear':
        candidate_Cs = [0.25, 0.5, 1.0, 2.0]
        best_C = C
        best_f1 = -1.0
        print("\nQuick tuning of C on validation split (linear kernel)...")
        for c_val in candidate_Cs:
            tmp_model = build_kernel_svm_pipeline(kernel=kernel, C=c_val, gamma=gamma)
            tmp_model.fit(X_tr, y_tr)
            tmp_pred = tmp_model.predict(X_val)
            tmp_f1 = f1_score(y_val, tmp_pred)
            print(f"  C={c_val:.2f} -> F1={tmp_f1:.4f}")
            if tmp_f1 > best_f1:
                best_f1 = tmp_f1
                best_C = c_val
        C = best_C
        print(f"Selected C={C:.2f} from quick search.")

    # Build final pipeline with chosen C
    pipeline = build_kernel_svm_pipeline(kernel=kernel, C=C, gamma=gamma)
    
    print(f"\nFitting Kernel SVM ({kernel.upper()} kernel) with C={C}...")
    if kernel == 'linear':
        print("Using LinearSVC (optimized for speed)...")
    else:
        print("Note: Non-linear kernels may take longer...")
    
    pipeline.fit(X_tr, y_tr)

    print("\n" + "-" * 80)
    print("VALIDATION METRICS")
    print("-" * 80)
    val_pred = pipeline.predict(X_val)

    print(f"Accuracy: {(val_pred == y_val).mean():.4f}")
    print(f"F1-score: {f1_score(y_val, val_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_pred))

    # Create visualizations before retraining on full dataset
    if create_plots:
        create_visualizations(pipeline, X_val, y_val, kernel, C)
    
    print("\nRetraining on the full dataset...")
    pipeline.fit(X_np, y_np)
    return pipeline, X_val, y_val


def make_predictions(model: Pipeline, X_test: pd.DataFrame):
    """Return class predictions and probabilities for the test set."""
    X_test_np = X_test.to_numpy(dtype=np.float32)
    preds = model.predict(X_test_np)
    
    # LinearSVC doesn't have predict_proba, so return None for probabilities
    proba = None
    if hasattr(model.named_steps['svm'], 'predict_proba'):
        try:
            proba = model.predict_proba(X_test_np)
        except:
            proba = None
    
    return preds, proba


def create_submission(
    patient_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = None,
):
    """Create submission matching sample_submission format."""
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "chronic-obstructive", "submission_kernel_svm.csv")
    
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
    
    # Train with LINEAR kernel and a sampled subset for much faster training
    # Linear kernel is much faster and often performs well; sampling cuts time further.
    model, X_val, y_val = train_kernel_svm(
        X_train, 
        y_train, 
        kernel='linear',  # Linear kernel is fastest
        C=1.0, 
        gamma='scale',
        use_sample=True,   # Use a subset to drastically reduce training time
        sample_size=8000,  # Adjust if you want even faster/slower training
        auto_tune_C=True,  # Do a tiny C search on the validation split
        create_plots=True,  # Generate visualization plots
    )
    
    predictions, _ = make_predictions(model, X_test)
    create_submission(patient_ids, predictions)


if __name__ == "__main__":
    main()

