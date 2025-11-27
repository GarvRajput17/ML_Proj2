# """
# K-Nearest Neighbors (KNN) Solution
# Configuration: K=5, Weighted by Distance.
# Target: High precision geometric classification.
# """

# import pandas as pd
# import numpy as np
# import warnings
# import os

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, f1_score

# warnings.filterwarnings('ignore')

# # ==========================================
# # CONFIGURATION
# # ==========================================
# ORIGINAL_TRAIN_FILE = 'train.csv'
# ORIGINAL_TEST_FILE  = 'test.csv'
# OUTPUT_FOLDER = 'output'

# def load_data():
#     """Load Raw Geometric Data"""
#     print("Loading data...")
#     paths = ['', 'data/']
#     X_train, X_test, y_train, test_ids = None, None, None, None
    
#     for p in paths:
#         try:
#             if y_train is None:
#                 train_df = pd.read_csv(os.path.join(p, ORIGINAL_TRAIN_FILE))
#                 test_df = pd.read_csv(os.path.join(p, ORIGINAL_TEST_FILE))
                
#                 # Select only the 2D coordinates
#                 X_train = train_df[['signal_strength', 'response_level']]
#                 y_train = train_df['category']
                
#                 X_test = test_df[['signal_strength', 'response_level']]
#                 test_ids = test_df['sample_id']
#         except FileNotFoundError:
#             continue
            
#     if X_train is None:
#         raise FileNotFoundError("Could not find train.csv/test.csv.")

#     # Clean NaNs
#     X_train = X_train.fillna(0)
#     X_test = X_test.fillna(0)

#     # Encode Target
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y_train)
    
#     print(f"Data Loaded: {X_train.shape} samples.")
#     return X_train, y_encoded, X_test, test_ids, le

# def get_knn_model():
#     print("\nConfiguring KNN (k=5)...")
    
#     # Pipeline: Scaling is MANDATORY for KNN
#     # weights='distance': Closer neighbors vote louder than far neighbors.
#     # p=2: Euclidean Distance (standard straight-line distance).
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('knn', KNeighborsClassifier(
#             n_neighbors=5, 
#             weights='distance', 
#             algorithm='auto',
#             p=2, 
#             n_jobs=-1
#         ))
#     ])
    
#     return pipeline

# def main():
#     print("\n" + "="*50)
#     print("TRAINING KNN (K=5)")
#     print("="*50)
    
#     # 1. Load Raw Data
#     X_train, y_train, X_test, test_ids, le = load_data()
    
#     # 2. Define Model
#     model = get_knn_model()
    
#     # 3. Validate
#     print("\nValidating (5-Fold CV)...")
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    
#     print(f"Fold Scores: {cv_scores}")
#     print(f"Mean Macro F1: {cv_scores.mean():.5f}")
    
#     # 4. Final Training
#     print("\nRetraining on FULL dataset...")
#     model.fit(X_train, y_train)
    
#     # 5. Predict
#     print("Generating Predictions...")
#     preds_encoded = model.predict(X_test)
#     preds_labels = le.inverse_transform(preds_encoded)
    
#     # 6. Save
#     if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
#     out_path = os.path.join(OUTPUT_FOLDER, 'submission_knn_k5.csv')
#     sub = pd.DataFrame({'sample_id': test_ids, 'category': preds_labels})
#     sub.to_csv(out_path, index=False)
    
#     print(f"\nSUCCESS! Saved to {out_path}")

# if __name__ == "__main__":
#     main()

"""
NCA-Optimized Bagged KNN
Constraint: K=5
Strategy: Learn a custom distance metric (NCA) to maximize class separation.
"""

import pandas as pd
import numpy as np
import warnings
import os

from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
ORIGINAL_TRAIN_FILE = 'train.csv'
ORIGINAL_TEST_FILE  = 'test.csv'
OUTPUT_FOLDER = 'output'

def load_data():
    """Load Raw Geometric Data"""
    print("Loading data...")
    paths = ['', 'data/']
    X_train, X_test, y_train, test_ids = None, None, None, None
    
    for p in paths:
        try:
            if y_train is None:
                train_df = pd.read_csv(os.path.join(p, ORIGINAL_TRAIN_FILE))
                test_df = pd.read_csv(os.path.join(p, ORIGINAL_TEST_FILE))
                
                # KNN needs Raw X, Y. No manual features.
                X_train = train_df[['signal_strength', 'response_level']]
                y_train = train_df['category']
                
                X_test = test_df[['signal_strength', 'response_level']]
                test_ids = test_df['sample_id']
        except FileNotFoundError:
            continue
            
    if X_train is None:
        raise FileNotFoundError("Could not find train.csv/test.csv.")

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    print(f"Data Loaded: {X_train.shape} samples.")
    return X_train, y_encoded, X_test, test_ids, le

def get_optimized_knn():
    print("\nConfiguring NCA-KNN (Metric Learning)...")
    
    # 1. THE PIPELINE
    # Step A: RobustScaler removes the influence of extreme outliers.
    # Step B: NeighborhoodComponentsAnalysis (NCA) learns the best metric.
    # Step C: KNN (k=5) uses that new metric.
    knn_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('nca', NeighborhoodComponentsAnalysis(random_state=42, init='pca')),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto'))
    ])
    
    # 2. BAGGING
    # We train 20 versions of this pipeline on different subsets of data.
    # This reduces the variance of the NCA transformation.
    bagged_knn = BaggingClassifier(
        estimator=knn_pipe,
        n_estimators=20,       # 20 different NCA transformations
        max_samples=0.8,       # Each sees 80% of data
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    return bagged_knn

def main():
    print("\n" + "="*50)
    print("TRAINING METRIC-LEARNING KNN (K=5)")
    print("="*50)
    
    # 1. Load
    X_train, y_train, X_test, test_ids, le = load_data()
    
    # 2. Define Model
    model = get_optimized_knn()
    
    # 3. Validate
    print("\nValidating (5-Fold CV)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    
    print(f"Fold Scores: {cv_scores}")
    print(f"Mean Macro F1: {cv_scores.mean():.5f}")
    
    if cv_scores.mean() > 0.985:
        print(">> STRONG GEOMETRIC PERFORMANCE DETECTED.")
    
    # 4. Final Training
    print("\nRetraining on FULL dataset...")
    model.fit(X_train, y_train)
    
    # 5. Predict
    print("Generating Predictions...")
    preds_encoded = model.predict(X_test)
    preds_labels = le.inverse_transform(preds_encoded)
    
    # 6. Save
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    out_path = os.path.join(OUTPUT_FOLDER, 'submission_nca_knn.csv')
    sub = pd.DataFrame({'sample_id': test_ids, 'category': preds_labels})
    sub.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS! Saved to {out_path}")
    print("This KNN learns the optimal shape of the clusters using NCA.")

if __name__ == "__main__":
    main()