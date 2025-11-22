"""
Comprehensive Data Exploration for Chronic Obstructive Pulmonary Disease (COPD) Risk Classification
This script performs exploratory data analysis on the training and test datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_data():
    """Load training and test datasets"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"\nTraining Data Shape: {train_df.shape}")
    print(f"Test Data Shape: {test_df.shape}")
    
    return train_df, test_df

def basic_info(train_df, test_df):
    """Display basic information about the datasets"""
    print("\n" + "=" * 80)
    print("BASIC DATASET INFORMATION")
    print("=" * 80)
    
    print("\n--- Training Data Info ---")
    print(train_df.info())
    
    print("\n--- Test Data Info ---")
    print(test_df.info())
    
    print("\n--- Training Data Head ---")
    print(train_df.head(10))
    
    print("\n--- Test Data Head ---")
    print(test_df.head(10))
    
    print("\n--- Training Data Columns ---")
    print(train_df.columns.tolist())
    
    print("\n--- Test Data Columns ---")
    print(test_df.columns.tolist())

def identify_feature_types(train_df):
    """Identify numerical and categorical features"""
    print("\n" + "=" * 80)
    print("FEATURE TYPE IDENTIFICATION")
    print("=" * 80)
    
    # Exclude patient_id and target
    exclude_cols = ['patient_id', 'has_copd_risk']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Identify numerical and categorical
    numerical_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also check for binary/encoded categoricals
    for col in feature_cols:
        if col not in numerical_cols and col not in categorical_cols:
            unique_vals = train_df[col].nunique()
            if unique_vals <= 10:  # Likely categorical
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
    
    print(f"\nNumerical Features ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"  - {col}")
    
    print(f"\nCategorical Features ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"  - {col} (unique values: {train_df[col].nunique()})")
    
    return numerical_cols, categorical_cols

def missing_values_analysis(train_df, test_df):
    """Analyze missing values in the datasets"""
    print("\n" + "=" * 80)
    print("MISSING VALUES ANALYSIS")
    print("=" * 80)
    
    print("\n--- Training Data Missing Values ---")
    missing_train = train_df.isnull().sum()
    missing_train = missing_train[missing_train > 0]
    if len(missing_train) > 0:
        print(missing_train)
    else:
        print("No missing values found!")
    print(f"\nTotal missing values in training: {train_df.isnull().sum().sum()}")
    
    print("\n--- Test Data Missing Values ---")
    missing_test = test_df.isnull().sum()
    missing_test = missing_test[missing_test > 0]
    if len(missing_test) > 0:
        print(missing_test)
    else:
        print("No missing values found!")
    print(f"\nTotal missing values in test: {test_df.isnull().sum().sum()}")

def target_variable_analysis(train_df):
    """Analyze the target variable (has_copd_risk)"""
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 80)
    
    print("\n--- Target Distribution ---")
    target_counts = train_df['has_copd_risk'].value_counts().sort_index()
    print(target_counts)
    
    print("\n--- Target Percentages ---")
    target_percentages = train_df['has_copd_risk'].value_counts(normalize=True).sort_index() * 100
    print(target_percentages)
    
    # Visualize target distribution
    plt.figure(figsize=(10, 6))
    target_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.title('Distribution of Target Variable (COPD Risk)', fontsize=14, fontweight='bold')
    plt.xlabel('Has COPD Risk (0=No, 1=Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: target_distribution.png")
    plt.close()

def descriptive_statistics(train_df, numerical_cols):
    """Compute descriptive statistics for numerical features"""
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    if len(numerical_cols) > 0:
        print("\n--- Training Data Statistics (Numerical Features) ---")
        print(train_df[numerical_cols].describe())
        
        print("\n--- Statistics by Target Variable ---")
        for target_val in sorted(train_df['has_copd_risk'].unique()):
            print(f"\nTarget = {target_val}:")
            subset = train_df[train_df['has_copd_risk'] == target_val][numerical_cols]
            print(subset.describe())

def categorical_analysis(train_df, categorical_cols):
    """Analyze categorical features"""
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("=" * 80)
    
    for col in categorical_cols:
        print(f"\n--- {col} ---")
        value_counts = train_df[col].value_counts()
        print(value_counts)
        
        # Cross-tabulation with target
        if col != 'has_copd_risk':
            print(f"\nCross-tabulation with target:")
            crosstab = pd.crosstab(train_df[col], train_df['has_copd_risk'], margins=True)
            print(crosstab)

def feature_distributions(train_df, numerical_cols):
    """Visualize distributions of numerical features"""
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 80)
    
    if len(numerical_cols) == 0:
        print("No numerical features to plot.")
        return
    
    # Select top 12 features for visualization
    n_features = min(12, len(numerical_cols))
    selected_features = numerical_cols[:n_features]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(selected_features):
        ax = axes[i]
        # Histogram
        train_df[feature].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_distributions.png")
    plt.close()

def box_plots_by_target(train_df, numerical_cols):
    """Create box plots for features by target variable"""
    print("\n" + "=" * 80)
    print("BOX PLOTS BY TARGET VARIABLE")
    print("=" * 80)
    
    if len(numerical_cols) == 0:
        print("No numerical features to plot.")
        return
    
    # Select top 12 features
    n_features = min(12, len(numerical_cols))
    selected_features = numerical_cols[:n_features]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, feature in enumerate(selected_features):
        ax = axes[i]
        train_df.boxplot(column=feature, by='has_copd_risk', ax=ax)
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Has COPD Risk')
        ax.set_ylabel('Value')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('box_plots_by_target.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: box_plots_by_target.png")
    plt.close()

def correlation_analysis(train_df, numerical_cols):
    """Analyze correlations between features"""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    if len(numerical_cols) == 0:
        print("No numerical features for correlation analysis.")
        return
    
    # Compute correlation matrix
    correlation_matrix = train_df[numerical_cols + ['has_copd_risk']].corr()
    
    # Correlation with target
    print("\n--- Correlation with Target Variable ---")
    target_corr = correlation_matrix['has_copd_risk'].sort_values(ascending=False)
    target_corr = target_corr[target_corr.index != 'has_copd_risk']
    print(target_corr)
    
    # Visualize correlation matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")
    plt.close()
    
    # Top correlations with target
    print("\n--- Top 10 Features Correlated with Target ---")
    top_corr = target_corr.abs().sort_values(ascending=False).head(10)
    print(top_corr)

def outlier_analysis(train_df, numerical_cols):
    """Detect and analyze outliers"""
    print("\n" + "=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)
    
    if len(numerical_cols) == 0:
        print("No numerical features for outlier analysis.")
        return
    
    outlier_summary = []
    
    for feature in numerical_cols:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = train_df[(train_df[feature] < lower_bound) | (train_df[feature] > upper_bound)]
        outlier_pct = len(outliers) / len(train_df) * 100
        
        outlier_summary.append({
            'Feature': feature,
            'Outliers': len(outliers),
            'Percentage': f"{outlier_pct:.2f}%",
            'Lower Bound': f"{lower_bound:.2f}",
            'Upper Bound': f"{upper_bound:.2f}"
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print("\nOutlier Summary (IQR Method):")
    print(outlier_df.to_string(index=False))

def train_test_comparison(train_df, test_df, numerical_cols):
    """Compare training and test data distributions"""
    print("\n" + "=" * 80)
    print("TRAIN-TEST DATA COMPARISON")
    print("=" * 80)
    
    if len(numerical_cols) == 0:
        print("No numerical features for comparison.")
        return
    
    print("\n--- Statistical Comparison (Top 10 Features) ---")
    top_features = numerical_cols[:10]
    
    for feature in top_features:
        if feature in test_df.columns:
            train_mean = train_df[feature].mean()
            test_mean = test_df[feature].mean()
            train_std = train_df[feature].std()
            test_std = test_df[feature].std()
            
            print(f"\n{feature}:")
            print(f"  Train - Mean: {train_mean:.2f}, Std: {train_std:.2f}")
            print(f"  Test  - Mean: {test_mean:.2f}, Std: {test_std:.2f}")
            print(f"  Difference in Mean: {abs(train_mean - test_mean):.2f}")

def generate_summary_report(train_df, test_df, numerical_cols, categorical_cols):
    """Generate a summary report"""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\nDataset Overview:")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Test samples: {len(test_df)}")
    print(f"  - Total features: {len(numerical_cols) + len(categorical_cols)}")
    print(f"    * Numerical: {len(numerical_cols)}")
    print(f"    * Categorical: {len(categorical_cols)}")
    print(f"  - Target variable: has_copd_risk (Binary Classification)")
    
    print(f"\nTarget Variable Distribution:")
    target_counts = train_df['has_copd_risk'].value_counts().sort_index()
    for val, count in target_counts.items():
        pct = count / len(train_df) * 100
        label = "Yes (Has Risk)" if val == 1 else "No (No Risk)"
        print(f"  - {label}: {count} ({pct:.2f}%)")
    
    print(f"\nMissing Values:")
    print(f"  - Training: {train_df.isnull().sum().sum()}")
    print(f"  - Test: {test_df.isnull().sum().sum()}")
    
    print(f"\nData Quality:")
    print(f"  - Duplicate samples in training: {train_df.duplicated().sum()}")
    print(f"  - Duplicate patient_ids in training: {train_df['patient_id'].duplicated().sum()}")
    
    if len(numerical_cols) > 0:
        print(f"\nNumerical Feature Ranges (Training):")
        for col in numerical_cols[:5]:  # Show first 5
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            print(f"  - {col}: [{min_val:.2f}, {max_val:.2f}]")

def main():
    """Main function to run all exploratory analyses"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATA EXPLORATION - COPD RISK PREDICTION")
    print("=" * 80)
    print("\nStarting data exploration...\n")
    
    # Load data
    train_df, test_df = load_data()
    
    # Identify feature types
    numerical_cols, categorical_cols = identify_feature_types(train_df)
    
    # Run all analyses
    basic_info(train_df, test_df)
    missing_values_analysis(train_df, test_df)
    target_variable_analysis(train_df)
    descriptive_statistics(train_df, numerical_cols)
    categorical_analysis(train_df, categorical_cols)
    feature_distributions(train_df, numerical_cols)
    box_plots_by_target(train_df, numerical_cols)
    correlation_analysis(train_df, numerical_cols)
    outlier_analysis(train_df, numerical_cols)
    train_test_comparison(train_df, test_df, numerical_cols)
    generate_summary_report(train_df, test_df, numerical_cols, categorical_cols)
    
    print("\n" + "=" * 80)
    print("DATA EXPLORATION COMPLETE!")
    print("=" * 80)
    print("\nAll visualizations have been saved as PNG files.")
    print("Review the output above and the generated plots for insights.\n")

if __name__ == "__main__":
    main()

