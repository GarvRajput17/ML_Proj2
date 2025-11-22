"""
Comprehensive Data Preprocessing and Feature Engineering
Creates linear, quadratic, logarithmic, and other feature transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class that creates various transformations:
    - Linear features (original)
    - Quadratic features (squared, interactions)
    - Logarithmic features
    - Other transformations (sqrt, exp, etc.)
    """
    
    def __init__(self, use_scaling=True, scaler_type='standard'):
        """
        Initialize the feature engineer
        
        Parameters:
        -----------
        use_scaling : bool, default=True
            Whether to apply scaling to features
        scaler_type : str, default='standard'
            Type of scaler: 'standard', 'minmax', 'robust', or None
        """
        self.use_scaling = use_scaling
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = []
        self.min_values = {}  # Store min values for log transformation
        
    def _create_linear_features(self, df):
        """Create linear features (original features)"""
        features = pd.DataFrame()
        features['signal_strength'] = df['signal_strength']
        features['response_level'] = df['response_level']
        return features
    
    def _create_quadratic_features(self, df):
        """Create quadratic features (squared terms and interactions)"""
        features = pd.DataFrame()
        
        # Squared terms
        features['signal_strength_squared'] = df['signal_strength'] ** 2
        features['response_level_squared'] = df['response_level'] ** 2
        
        # Interaction terms
        features['signal_response_interaction'] = df['signal_strength'] * df['response_level']
        
        # Ratio features
        features['signal_response_ratio'] = df['signal_strength'] / (df['response_level'] + 1e-8)
        features['response_signal_ratio'] = df['response_level'] / (df['signal_strength'] + 1e-8)
        
        # Difference and sum
        features['signal_response_diff'] = df['signal_strength'] - df['response_level']
        features['signal_response_sum'] = df['signal_strength'] + df['response_level']
        
        # Absolute difference
        features['signal_response_abs_diff'] = np.abs(df['signal_strength'] - df['response_level'])
        
        return features
    
    def _create_logarithmic_features(self, df):
        """Create logarithmic features with handling for negative values"""
        features = pd.DataFrame()
        
        # Store minimum values for shifting (to handle negative values)
        if 'signal_strength' not in self.min_values:
            self.min_values['signal_strength'] = df['signal_strength'].min()
            self.min_values['response_level'] = df['response_level'].min()
        
        # Shift to make all values positive for log transformation
        signal_shifted = df['signal_strength'] - self.min_values['signal_strength'] + 1
        response_shifted = df['response_level'] - self.min_values['response_level'] + 1
        
        # Natural logarithm
        features['log_signal_strength'] = np.log1p(signal_shifted)
        features['log_response_level'] = np.log1p(response_shifted)
        
        # Log10
        features['log10_signal_strength'] = np.log10(signal_shifted)
        features['log10_response_level'] = np.log10(response_shifted)
        
        # Log of squared terms
        features['log_signal_squared'] = np.log1p(signal_shifted ** 2)
        features['log_response_squared'] = np.log1p(response_shifted ** 2)
        
        # Log of interaction
        features['log_signal_response_interaction'] = np.log1p(signal_shifted * response_shifted)
        
        return features
    
    def _create_other_transformations(self, df):
        """Create other mathematical transformations"""
        features = pd.DataFrame()
        
        # Square root (shift to handle negative values)
        signal_shifted = df['signal_strength'] - self.min_values.get('signal_strength', df['signal_strength'].min()) + 1
        response_shifted = df['response_level'] - self.min_values.get('response_level', df['response_level'].min()) + 1
        
        features['sqrt_signal_strength'] = np.sqrt(np.abs(signal_shifted))
        features['sqrt_response_level'] = np.sqrt(np.abs(response_shifted))
        
        # Exponential (normalized to avoid overflow)
        features['exp_signal_normalized'] = np.exp(df['signal_strength'] / 100)
        features['exp_response_normalized'] = np.exp(df['response_level'] / 100)
        
        # Trigonometric transformations (normalized)
        features['sin_signal'] = np.sin(df['signal_strength'] / 100)
        features['cos_signal'] = np.cos(df['signal_strength'] / 100)
        features['sin_response'] = np.sin(df['response_level'] / 100)
        features['cos_response'] = np.cos(df['response_level'] / 100)
        
        # Power transformations
        features['signal_power_1_5'] = np.power(np.abs(signal_shifted), 1.5)
        features['response_power_1_5'] = np.power(np.abs(response_shifted), 1.5)
        features['signal_power_0_5'] = np.power(np.abs(signal_shifted), 0.5)
        features['response_power_0_5'] = np.power(np.abs(response_shifted), 0.5)
        
        # Reciprocal (with small epsilon to avoid division by zero)
        epsilon = 1e-8
        features['reciprocal_signal'] = 1 / (np.abs(df['signal_strength']) + epsilon)
        features['reciprocal_response'] = 1 / (np.abs(df['response_level']) + epsilon)
        
        return features
    
    def _create_statistical_features(self, df):
        """Create statistical features"""
        features = pd.DataFrame()
        
        # Normalized features (z-score like, but using min-max)
        signal_min, signal_max = df['signal_strength'].min(), df['signal_strength'].max()
        response_min, response_max = df['response_level'].min(), df['response_level'].max()
        
        features['signal_normalized'] = (df['signal_strength'] - signal_min) / (signal_max - signal_min + 1e-8)
        features['response_normalized'] = (df['response_level'] - response_min) / (response_max - response_min + 1e-8)
        
        # Distance from mean
        signal_mean = df['signal_strength'].mean()
        response_mean = df['response_level'].mean()
        
        features['signal_distance_from_mean'] = np.abs(df['signal_strength'] - signal_mean)
        features['response_distance_from_mean'] = np.abs(df['response_level'] - response_mean)
        
        return features
    
    def fit_transform(self, df, feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical']):
        """
        Fit and transform the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with 'signal_strength' and 'response_level' columns
        feature_types : list
            List of feature types to create: 'linear', 'quadratic', 'logarithmic', 'other', 'statistical'
        
        Returns:
        --------
        pd.DataFrame: Transformed features
        """
        all_features = []
        
        # Create different feature types
        if 'linear' in feature_types:
            linear_features = self._create_linear_features(df)
            all_features.append(linear_features)
            print(f"✓ Created {len(linear_features.columns)} linear features")
        
        if 'quadratic' in feature_types:
            quadratic_features = self._create_quadratic_features(df)
            all_features.append(quadratic_features)
            print(f"✓ Created {len(quadratic_features.columns)} quadratic features")
        
        if 'logarithmic' in feature_types:
            logarithmic_features = self._create_logarithmic_features(df)
            all_features.append(logarithmic_features)
            print(f"✓ Created {len(logarithmic_features.columns)} logarithmic features")
        
        if 'other' in feature_types:
            other_features = self._create_other_transformations(df)
            all_features.append(other_features)
            print(f"✓ Created {len(other_features.columns)} other transformation features")
        
        if 'statistical' in feature_types:
            statistical_features = self._create_statistical_features(df)
            all_features.append(statistical_features)
            print(f"✓ Created {len(statistical_features.columns)} statistical features")
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
        else:
            combined_features = pd.DataFrame()
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        
        # Apply scaling if requested
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
    
    def transform(self, df, feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical']):
        """
        Transform new data using fitted parameters
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with 'signal_strength' and 'response_level' columns
        feature_types : list
            List of feature types to create
        
        Returns:
        --------
        pd.DataFrame: Transformed features
        """
        all_features = []
        
        # Create different feature types
        if 'linear' in feature_types:
            linear_features = self._create_linear_features(df)
            all_features.append(linear_features)
        
        if 'quadratic' in feature_types:
            quadratic_features = self._create_quadratic_features(df)
            all_features.append(quadratic_features)
        
        if 'logarithmic' in feature_types:
            logarithmic_features = self._create_logarithmic_features(df)
            all_features.append(logarithmic_features)
        
        if 'other' in feature_types:
            other_features = self._create_other_transformations(df)
            all_features.append(other_features)
        
        if 'statistical' in feature_types:
            statistical_features = self._create_statistical_features(df)
            all_features.append(statistical_features)
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
        else:
            combined_features = pd.DataFrame()
        
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
                   feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical'],
                   use_scaling=True, scaler_type='standard',
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
    save_processed : bool
        Whether to save processed data to CSV files
    
    Returns:
    --------
    tuple: (X_train, y_train, X_test, feature_engineer)
    """
    print("=" * 80)
    print("DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Separate features and target
    X_train_raw = train_df[['signal_strength', 'response_level']]
    y_train = train_df['category']
    X_test_raw = test_df[['signal_strength', 'response_level']]
    
    # Initialize feature engineer
    print("\nCreating features...")
    feature_engineer = FeatureEngineer(use_scaling=use_scaling, scaler_type=scaler_type)
    
    # Transform training data
    X_train = feature_engineer.fit_transform(X_train_raw, feature_types=feature_types)
    
    # Transform test data
    X_test = feature_engineer.transform(X_test_raw, feature_types=feature_types)
    
    print(f"\n✓ Final feature count: {X_train.shape[1]}")
    print(f"✓ Training features shape: {X_train.shape}")
    print(f"✓ Test features shape: {X_test.shape}")
    
    # Save processed data if requested
    if save_processed:
        X_train.to_csv('processed_train_features.csv', index=False)
        X_test.to_csv('processed_test_features.csv', index=False)
        y_train.to_csv('train_target.csv', index=False)
        print("\n✓ Saved processed features to CSV files")
    
    return X_train, y_train, X_test, feature_engineer


def get_feature_summary(feature_engineer):
    """Get summary of created features"""
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    print(f"\nTotal features created: {len(feature_engineer.feature_names)}")
    print("\nFeature list:")
    for i, feature in enumerate(feature_engineer.feature_names, 1):
        print(f"  {i:2d}. {feature}")


if __name__ == "__main__":
    # Example usage
    train_path = 'Single_cluster_classification/train.csv'
    test_path = 'Single_cluster_classification/test.csv'
    
    # Process data with all feature types
    X_train, y_train, X_test, fe = preprocess_data(
        train_path=train_path,
        test_path=test_path,
        feature_types=['linear', 'quadratic', 'logarithmic', 'other', 'statistical'],
        use_scaling=True,
        scaler_type='standard',
        save_processed=True
    )
    
    # Display feature summary
    get_feature_summary(fe)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)

