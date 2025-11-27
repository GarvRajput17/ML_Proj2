"""
Comprehensive Data Preprocessing and Feature Engineering for COPD Risk Prediction
Handles both numerical and categorical features with various transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for COPD dataset that creates various transformations:
    - Linear features (original)
    - Quadratic features (squared, interactions)
    - Logarithmic features
    - Other transformations (sqrt, exp, etc.)
    - Categorical encoding
    """
    
    def __init__(self, use_scaling=True, scaler_type='standard', 
                 categorical_encoding='onehot'):
        """
        Initialize the feature engineer
        
        Parameters:
        -----------
        use_scaling : bool, default=True
            Whether to apply scaling to numerical features
        scaler_type : str, default='standard'
            Type of scaler: 'standard', 'minmax', 'robust', or None
        categorical_encoding : str, default='onehot'
            Type of categorical encoding: 'onehot', 'label', or 'both'
        """
        self.use_scaling = use_scaling
        self.scaler_type = scaler_type
        self.categorical_encoding = categorical_encoding
        self.scaler = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.feature_names = []
        self.numerical_cols = []
        self.categorical_cols = []
        self.min_values = {}  # Store min values for log transformation
        
    def _identify_feature_types(self, df):
        """Identify numerical and categorical features"""
        exclude_cols = ['patient_id', 'has_copd_risk']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for binary/encoded categoricals
        for col in feature_cols:
            if col not in numerical_cols and col not in categorical_cols:
                unique_vals = df[col].nunique()
                if unique_vals <= 10:  # Likely categorical
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        
        return numerical_cols, categorical_cols
    
    def _create_linear_features(self, df, numerical_cols):
        """Create linear features (original numerical features)"""
        features = pd.DataFrame()
        for col in numerical_cols:
            if col in df.columns:
                features[col] = df[col]
        return features
    
    def _create_quadratic_features(self, df, numerical_cols):
        """Create quadratic features (squared terms and interactions)"""
        features = pd.DataFrame()
        
        # Squared terms
        for col in numerical_cols:
            if col in df.columns:
                features[f'{col}_squared'] = df[col] ** 2
        
        # Interaction terms (top features only to avoid explosion)
        top_features = numerical_cols[:10]  # Limit to top 10 to avoid too many features
        for i, col1 in enumerate(top_features):
            for col2 in top_features[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    features[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        # Key ratio features (important health metrics)
        if 'weight_kg' in df.columns and 'height_cm' in df.columns:
            features['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        
        if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
            features['bp_mean'] = (df['bp_systolic'] + df['bp_diastolic']) / 2
            features['bp_pulse_pressure'] = df['bp_systolic'] - df['bp_diastolic']
        
        if 'hdl_cholesterol' in df.columns and 'ldl_cholesterol' in df.columns:
            features['cholesterol_ratio'] = df['hdl_cholesterol'] / (df['ldl_cholesterol'] + 1e-8)
        
        if 'total_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
            features['cholesterol_hdl_ratio'] = df['total_cholesterol'] / (df['hdl_cholesterol'] + 1e-8)
        
        # Vision and hearing averages
        if 'vision_left' in df.columns and 'vision_right' in df.columns:
            features['vision_avg'] = (df['vision_left'] + df['vision_right']) / 2
            features['vision_diff'] = abs(df['vision_left'] - df['vision_right'])
        
        if 'hearing_left' in df.columns and 'hearing_right' in df.columns:
            features['hearing_avg'] = (df['hearing_left'] + df['hearing_right']) / 2
        
        # Enzyme ratios
        if 'ast_enzyme_level' in df.columns and 'alt_enzyme_level' in df.columns:
            features['ast_alt_ratio'] = df['ast_enzyme_level'] / (df['alt_enzyme_level'] + 1e-8)
        
        return features
    
    def _create_logarithmic_features(self, df, numerical_cols):
        """Create logarithmic features with handling for negative values"""
        features = pd.DataFrame()
        
        # Select key features for log transformation (to avoid too many features)
        key_features = [col for col in numerical_cols if col in df.columns][:15]
        
        for col in key_features:
            # Store minimum values for shifting
            if col not in self.min_values:
                self.min_values[col] = df[col].min()
            
            # Shift to make all values positive for log transformation
            shifted = df[col] - self.min_values[col] + 1
            
            # Natural logarithm
            features[f'log_{col}'] = np.log1p(shifted)
            
            # Log10 for positive values
            if (shifted > 0).all():
                features[f'log10_{col}'] = np.log10(shifted)
        
        return features
    
    def _create_other_transformations(self, df, numerical_cols):
        """Create other mathematical transformations"""
        features = pd.DataFrame()
        
        # Select key features
        key_features = [col for col in numerical_cols if col in df.columns][:10]
        
        for col in key_features:
            # Shift to handle negative values
            if col not in self.min_values:
                self.min_values[col] = df[col].min()
            shifted = df[col] - self.min_values[col] + 1
            
            # Square root
            features[f'sqrt_{col}'] = np.sqrt(np.abs(shifted))
            
            # Power transformations
            features[f'{col}_power_1_5'] = np.power(np.abs(shifted), 1.5)
            features[f'{col}_power_0_5'] = np.power(np.abs(shifted), 0.5)
            
            # Normalized exponential (to avoid overflow)
            if df[col].std() > 0:
                normalized = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                features[f'exp_{col}_normalized'] = np.exp(normalized / 3)  # Divide by 3 to prevent overflow
        
        return features
    
    def _create_statistical_features(self, df, numerical_cols):
        """Create statistical features"""
        features = pd.DataFrame()
        
        # Select all numerical features for statistical transformations
        for col in numerical_cols:
            if col in df.columns:
                # Normalized features (min-max)
                col_min, col_max = df[col].min(), df[col].max()
                if col_max > col_min:
                    features[f'{col}_normalized'] = (df[col] - col_min) / (col_max - col_min + 1e-8)
                
                # Distance from mean
                col_mean = df[col].mean()
                features[f'{col}_distance_from_mean'] = np.abs(df[col] - col_mean)
                
                # Z-score
                col_std = df[col].std()
                if col_std > 0:
                    features[f'{col}_zscore'] = (df[col] - col_mean) / col_std
        
        return features
    
    def _encode_categorical_features(self, df, categorical_cols, is_training=True):
        """Encode categorical features"""
        features = pd.DataFrame()
        
        if len(categorical_cols) == 0:
            return features
        
        # Prepare categorical data
        cat_data = df[categorical_cols].copy()
        
        if self.categorical_encoding in ['onehot', 'both']:
            # One-hot encoding
            if is_training:
                # Fit one-hot encoder
                from sklearn.preprocessing import OneHotEncoder
                self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = self.onehot_encoder.fit_transform(cat_data)
                feature_names = self.onehot_encoder.get_feature_names_out(categorical_cols)
            else:
                # Transform using fitted encoder
                if self.onehot_encoder is not None:
                    encoded = self.onehot_encoder.transform(cat_data)
                    feature_names = self.onehot_encoder.get_feature_names_out(categorical_cols)
                else:
                    encoded = np.array([])
                    feature_names = []
            
            # Add one-hot encoded features
            for i, name in enumerate(feature_names):
                features[name] = encoded[:, i]
        
        if self.categorical_encoding in ['label', 'both']:
            # Label encoding
            for col in categorical_cols:
                if col in df.columns:
                    if is_training:
                        le = LabelEncoder()
                        features[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            # Handle unseen categories
                            unique_vals = set(le.classes_)
                            encoded_vals = df[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in unique_vals else -1
                            )
                            features[f'{col}_encoded'] = encoded_vals
                        else:
                            features[f'{col}_encoded'] = 0
        
        return features
    
    def fit_transform(self, df, feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical', 'categorical']):
        """
        Fit and transform the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_types : list
            List of feature types to create
        
        Returns:
        --------
        pd.DataFrame: Transformed features
        """
        all_features = []
        
        # Identify feature types
        self.numerical_cols, self.categorical_cols = self._identify_feature_types(df)
        
        print(f"Identified {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} categorical features")
        
        # Create different feature types for numerical features
        if 'linear' in feature_types:
            linear_features = self._create_linear_features(df, self.numerical_cols)
            if not linear_features.empty:
                all_features.append(linear_features)
                print(f"✓ Created {len(linear_features.columns)} linear features")
        
        if 'quadratic' in feature_types:
            quadratic_features = self._create_quadratic_features(df, self.numerical_cols)
            if not quadratic_features.empty:
                all_features.append(quadratic_features)
                print(f"✓ Created {len(quadratic_features.columns)} quadratic features")
        
        if 'logarithmic' in feature_types:
            logarithmic_features = self._create_logarithmic_features(df, self.numerical_cols)
            if not logarithmic_features.empty:
                all_features.append(logarithmic_features)
                print(f"✓ Created {len(logarithmic_features.columns)} logarithmic features")
        
        if 'other' in feature_types:
            other_features = self._create_other_transformations(df, self.numerical_cols)
            if not other_features.empty:
                all_features.append(other_features)
                print(f"✓ Created {len(other_features.columns)} other transformation features")
        
        if 'statistical' in feature_types:
            statistical_features = self._create_statistical_features(df, self.numerical_cols)
            if not statistical_features.empty:
                all_features.append(statistical_features)
                print(f"✓ Created {len(statistical_features.columns)} statistical features")
        
        # Encode categorical features
        if 'categorical' in feature_types:
            categorical_features = self._encode_categorical_features(df, self.categorical_cols, is_training=True)
            if not categorical_features.empty:
                all_features.append(categorical_features)
                print(f"✓ Created {len(categorical_features.columns)} categorical encoded features")
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
        else:
            combined_features = pd.DataFrame()
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        
        # Apply scaling if requested (only to numerical features)
        if self.use_scaling and len(combined_features) > 0:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = None
            
            if self.scaler is not None:
                scaled_data = self.scaler.fit_transform(combined_features)
                combined_features = pd.DataFrame(
                    scaled_data, 
                    columns=combined_features.columns,
                    index=combined_features.index
                )
                print(f"✓ Applied {self.scaler_type} scaling")
        
        return combined_features
    
    def transform(self, df, feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical', 'categorical']):
        """
        Transform new data using fitted parameters
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_types : list
            List of feature types to create
        
        Returns:
        --------
        pd.DataFrame: Transformed features
        """
        all_features = []
        
        # Create different feature types for numerical features
        if 'linear' in feature_types:
            linear_features = self._create_linear_features(df, self.numerical_cols)
            if not linear_features.empty:
                all_features.append(linear_features)
        
        if 'quadratic' in feature_types:
            quadratic_features = self._create_quadratic_features(df, self.numerical_cols)
            if not quadratic_features.empty:
                all_features.append(quadratic_features)
        
        if 'logarithmic' in feature_types:
            logarithmic_features = self._create_logarithmic_features(df, self.numerical_cols)
            if not logarithmic_features.empty:
                all_features.append(logarithmic_features)
        
        if 'other' in feature_types:
            other_features = self._create_other_transformations(df, self.numerical_cols)
            if not other_features.empty:
                all_features.append(other_features)
        
        if 'statistical' in feature_types:
            statistical_features = self._create_statistical_features(df, self.numerical_cols)
            if not statistical_features.empty:
                all_features.append(statistical_features)
        
        # Encode categorical features
        if 'categorical' in feature_types:
            categorical_features = self._encode_categorical_features(df, self.categorical_cols, is_training=False)
            if not categorical_features.empty:
                all_features.append(categorical_features)
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
        else:
            combined_features = pd.DataFrame()
        
        # Ensure same columns as training
        if len(self.feature_names) > 0:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in combined_features.columns:
                    combined_features[col] = 0
            
            # Remove extra columns
            combined_features = combined_features[self.feature_names]
        
        # Apply scaling if fitted
        if self.scaler is not None:
            scaled_data = self.scaler.transform(combined_features)
            combined_features = pd.DataFrame(
                scaled_data, 
                columns=combined_features.columns,
                index=combined_features.index
            )
        
        return combined_features


def preprocess_data(train_path, test_path, 
                   feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical', 'categorical'],
                   use_scaling=True, scaler_type='standard',
                   categorical_encoding='onehot',
                   save_processed=False):
    """
    Main preprocessing function
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
    feature_types : list
        Types of features to create
    use_scaling : bool
        Whether to apply scaling
    scaler_type : str
        Type of scaler to use
    categorical_encoding : str
        Type of categorical encoding
    save_processed : bool
        Whether to save processed data to CSV files
    
    Returns:
    --------
    tuple: (X_train, y_train, X_test, feature_engineer)
    """
    print("=" * 80)
    print("DATA PREPROCESSING AND FEATURE ENGINEERING - COPD RISK PREDICTION")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Separate features and target
    y_train = train_df['has_copd_risk'] if 'has_copd_risk' in train_df.columns else None
    
    # Initialize feature engineer
    print("\nCreating features...")
    feature_engineer = FeatureEngineer(
        use_scaling=use_scaling, 
        scaler_type=scaler_type,
        categorical_encoding=categorical_encoding
    )
    
    # Transform training data
    X_train = feature_engineer.fit_transform(train_df, feature_types=feature_types)
    
    # Transform test data
    X_test = feature_engineer.transform(test_df, feature_types=feature_types)
    
    print(f"\n✓ Final feature count: {X_train.shape[1]}")
    print(f"✓ Training features shape: {X_train.shape}")
    print(f"✓ Test features shape: {X_test.shape}")
    
    # Save processed data if requested
    if save_processed:
        X_train.to_csv('processed_train_features.csv', index=False)
        X_test.to_csv('processed_test_features.csv', index=False)
        if y_train is not None:
            y_train.to_csv('train_target.csv', index=False)
        print("\n✓ Saved processed features to CSV files")
    
    return X_train, y_train, X_test, feature_engineer


def get_feature_summary(feature_engineer):
    """Get summary of created features"""
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    print(f"\nTotal features created: {len(feature_engineer.feature_names)}")
    print(f"Numerical features: {len(feature_engineer.numerical_cols)}")
    print(f"Categorical features: {len(feature_engineer.categorical_cols)}")
    print("\nFeature list (first 50):")
    for i, feature in enumerate(feature_engineer.feature_names[:50], 1):
        print(f"  {i:2d}. {feature}")
    if len(feature_engineer.feature_names) > 50:
        print(f"  ... and {len(feature_engineer.feature_names) - 50} more features")


if __name__ == "__main__":
    # Example usage
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    # Process data with all feature types
    X_train, y_train, X_test, fe = preprocess_data(
        train_path=train_path,
        test_path=test_path,
        feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical', 'categorical'],
        use_scaling=True,
        scaler_type='standard',
        categorical_encoding='onehot',
        save_processed=True
    )
    
    # Display feature summary
    get_feature_summary(fe)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)

