"""
Comprehensive Data Exploration for Classification Task
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
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load training and test datasets"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_df = pd.read_csv('Single_cluster_classification/train.csv')
    test_df = pd.read_csv('Single_cluster_classification/test.csv')
    
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

def missing_values_analysis(train_df, test_df):
    """Analyze missing values in the datasets"""
    print("\n" + "=" * 80)
    print("MISSING VALUES ANALYSIS")
    print("=" * 80)
    
    print("\n--- Training Data Missing Values ---")
    missing_train = train_df.isnull().sum()
    print(missing_train)
    print(f"\nTotal missing values in training: {missing_train.sum()}")
    
    print("\n--- Test Data Missing Values ---")
    missing_test = test_df.isnull().sum()
    print(missing_test)
    print(f"\nTotal missing values in test: {missing_test.sum()}")

def target_variable_analysis(train_df):
    """Analyze the target variable (category)"""
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 80)
    
    print("\n--- Category Distribution ---")
    category_counts = train_df['category'].value_counts()
    print(category_counts)
    
    print("\n--- Category Percentages ---")
    category_percentages = train_df['category'].value_counts(normalize=True) * 100
    print(category_percentages)
    
    # Visualize target distribution
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.title('Distribution of Target Variable (Category)', fontsize=14, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: target_distribution.png")
    plt.close()

def descriptive_statistics(train_df, test_df):
    """Compute descriptive statistics for numerical features"""
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    numerical_features = ['signal_strength', 'response_level']
    
    print("\n--- Training Data Statistics ---")
    print(train_df[numerical_features].describe())
    
    print("\n--- Test Data Statistics ---")
    print(test_df[numerical_features].describe())
    
    print("\n--- Statistics by Category (Training) ---")
    for category in train_df['category'].unique():
        print(f"\n{category}:")
        print(train_df[train_df['category'] == category][numerical_features].describe())

def feature_distributions(train_df):
    """Visualize distributions of features"""
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Signal Strength Distribution
    axes[0, 0].hist(train_df['signal_strength'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Signal Strength Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Signal Strength')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(alpha=0.3)
    
    # Response Level Distribution
    axes[0, 1].hist(train_df['response_level'], bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Response Level Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Response Level')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)
    
    # Signal Strength by Category
    for category in train_df['category'].unique():
        axes[1, 0].hist(train_df[train_df['category'] == category]['signal_strength'], 
                       bins=30, alpha=0.6, label=category, edgecolor='black')
    axes[1, 0].set_title('Signal Strength Distribution by Category', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Signal Strength')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Response Level by Category
    for category in train_df['category'].unique():
        axes[1, 1].hist(train_df[train_df['category'] == category]['response_level'], 
                       bins=30, alpha=0.6, label=category, edgecolor='black')
    axes[1, 1].set_title('Response Level Distribution by Category', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Response Level')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_distributions.png")
    plt.close()

def box_plots(train_df):
    """Create box plots for features by category"""
    print("\n" + "=" * 80)
    print("BOX PLOTS ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot for Signal Strength
    sns.boxplot(data=train_df, x='category', y='signal_strength', ax=axes[0])
    axes[0].set_title('Signal Strength by Category', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Signal Strength')
    axes[0].grid(alpha=0.3)
    
    # Box plot for Response Level
    sns.boxplot(data=train_df, x='category', y='response_level', ax=axes[1])
    axes[1].set_title('Response Level by Category', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Response Level')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('box_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: box_plots.png")
    plt.close()

def scatter_plot_analysis(train_df):
    """Create scatter plots to visualize relationships"""
    print("\n" + "=" * 80)
    print("SCATTER PLOT ANALYSIS")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot colored by category
    categories = train_df['category'].unique()
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, category in enumerate(categories):
        mask = train_df['category'] == category
        axes[0].scatter(train_df[mask]['signal_strength'], 
                       train_df[mask]['response_level'],
                       label=category, alpha=0.6, s=30, color=colors[i])
    
    axes[0].set_title('Signal Strength vs Response Level (by Category)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Signal Strength')
    axes[0].set_ylabel('Response Level')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Density plot
    for category in categories:
        mask = train_df['category'] == category
        axes[1].scatter(train_df[mask]['signal_strength'], 
                       train_df[mask]['response_level'],
                       label=category, alpha=0.4, s=20)
    
    axes[1].set_title('Signal Strength vs Response Level (Overlay)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Signal Strength')
    axes[1].set_ylabel('Response Level')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: scatter_plots.png")
    plt.close()

def correlation_analysis(train_df):
    """Analyze correlations between features"""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Numerical correlation
    numerical_df = train_df[['signal_strength', 'response_level']]
    correlation_matrix = numerical_df.corr()
    
    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)
    
    # Visualize correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")
    plt.close()

def outlier_analysis(train_df):
    """Detect and analyze outliers"""
    print("\n" + "=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)
    
    numerical_features = ['signal_strength', 'response_level']
    
    for feature in numerical_features:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = train_df[(train_df[feature] < lower_bound) | (train_df[feature] > upper_bound)]
        
        print(f"\n--- {feature} Outliers (IQR Method) ---")
        print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(train_df)*100:.2f}%)")
        
        if len(outliers) > 0:
            print(f"Outlier range: [{outliers[feature].min():.2f}, {outliers[feature].max():.2f}]")

def feature_statistics_by_category(train_df):
    """Detailed statistics for each feature by category"""
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS BY CATEGORY")
    print("=" * 80)
    
    numerical_features = ['signal_strength', 'response_level']
    
    for feature in numerical_features:
        print(f"\n--- {feature} Statistics by Category ---")
        stats_by_category = train_df.groupby('category')[feature].agg([
            'count', 'mean', 'std', 'min', 'max', 'median', 
            lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
        ])
        stats_by_category.columns = ['Count', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Q1', 'Q3']
        print(stats_by_category)

def train_test_comparison(train_df, test_df):
    """Compare training and test data distributions"""
    print("\n" + "=" * 80)
    print("TRAIN-TEST DATA COMPARISON")
    print("=" * 80)
    
    numerical_features = ['signal_strength', 'response_level']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, feature in enumerate(numerical_features):
        axes[i].hist(train_df[feature], bins=50, alpha=0.6, label='Train', color='blue', edgecolor='black')
        axes[i].hist(test_df[feature], bins=50, alpha=0.6, label='Test', color='red', edgecolor='black')
        axes[i].set_title(f'{feature} Distribution: Train vs Test', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('train_test_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: train_test_comparison.png")
    plt.close()
    
    # Statistical comparison
    print("\n--- Statistical Comparison ---")
    for feature in numerical_features:
        print(f"\n{feature}:")
        print(f"  Train - Mean: {train_df[feature].mean():.2f}, Std: {train_df[feature].std():.2f}")
        print(f"  Test  - Mean: {test_df[feature].mean():.2f}, Std: {test_df[feature].std():.2f}")
        print(f"  Difference in Mean: {abs(train_df[feature].mean() - test_df[feature].mean()):.2f}")

def generate_summary_report(train_df, test_df):
    """Generate a summary report"""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\nDataset Overview:")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Test samples: {len(test_df)}")
    print(f"  - Features: {len(train_df.columns) - 2} (excluding sample_id and target)")
    print(f"  - Target classes: {train_df['category'].nunique()} ({', '.join(train_df['category'].unique())})")
    
    print(f"\nTarget Variable Distribution:")
    for category, count in train_df['category'].value_counts().items():
        print(f"  - {category}: {count} ({count/len(train_df)*100:.2f}%)")
    
    print(f"\nFeature Ranges (Training):")
    print(f"  - Signal Strength: [{train_df['signal_strength'].min():.2f}, {train_df['signal_strength'].max():.2f}]")
    print(f"  - Response Level: [{train_df['response_level'].min():.2f}, {train_df['response_level'].max():.2f}]")
    
    print(f"\nMissing Values:")
    print(f"  - Training: {train_df.isnull().sum().sum()}")
    print(f"  - Test: {test_df.isnull().sum().sum()}")
    
    print(f"\nData Quality:")
    print(f"  - Duplicate samples in training: {train_df.duplicated().sum()}")
    print(f"  - Duplicate sample_ids in training: {train_df['sample_id'].duplicated().sum()}")

def main():
    """Main function to run all exploratory analyses"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATA EXPLORATION")
    print("=" * 80)
    print("\nStarting data exploration...\n")
    
    # Load data
    train_df, test_df = load_data()
    
    # Run all analyses
    basic_info(train_df, test_df)
    missing_values_analysis(train_df, test_df)
    target_variable_analysis(train_df)
    descriptive_statistics(train_df, test_df)
    feature_distributions(train_df)
    box_plots(train_df)
    scatter_plot_analysis(train_df)
    correlation_analysis(train_df)
    outlier_analysis(train_df)
    feature_statistics_by_category(train_df)
    train_test_comparison(train_df, test_df)
    generate_summary_report(train_df, test_df)
    
    print("\n" + "=" * 80)
    print("DATA EXPLORATION COMPLETE!")
    print("=" * 80)
    print("\nAll visualizations have been saved as PNG files.")
    print("Review the output above and the generated plots for insights.\n")

if __name__ == "__main__":
    main()

