"""
Exploratory Data Analysis utilities for house price prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import warnings

from config.settings import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


class EDAAnalyzer:
    """Comprehensive EDA analysis class for house price data."""

    def __init__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None):
        """
        Initialize EDA analyzer.

        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
        """
        self.train_data = train_data.copy()
        self.test_data = test_data.copy() if test_data is not None else None
        self.target_col = 'SalePrice'

        # Store original data for reference
        self.original_train_shape = self.train_data.shape
        self.original_test_shape = self.test_data.shape if self.test_data is not None else None

    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.

        Returns:
            DataFrame with missing value statistics
        """
        missing_train = self.train_data.isnull().sum()
        missing_pct_train = (missing_train / len(self.train_data)) * 100

        result = pd.DataFrame({
            'Train_Missing_Count': missing_train,
            'Train_Missing_Percentage': missing_pct_train
        })

        if self.test_data is not None:
            missing_test = self.test_data.isnull().sum()
            missing_pct_test = (missing_test / len(self.test_data)) * 100
            result['Test_Missing_Count'] = missing_test
            result['Test_Missing_Percentage'] = missing_pct_test

        return result.sort_values('Train_Missing_Count', ascending=False)

    def analyze_numerical_features(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze numerical features in the dataset.

        Returns:
            Dictionary with numerical feature statistics
        """
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)  # Exclude target from feature analysis

        stats = {}
        for col in numerical_cols:
            if col in self.train_data.columns:
                stats[col] = {
                    'mean': float(self.train_data[col].mean()),
                    'median': float(self.train_data[col].median()),
                    'std': float(self.train_data[col].std()),
                    'min': float(self.train_data[col].min()),
                    'max': float(self.train_data[col].max()),
                    'skewness': float(self.train_data[col].skew()),
                    'kurtosis': float(self.train_data[col].kurtosis()),
                    'missing_count': int(self.train_data[col].isnull().sum()),
                    'unique_values': int(self.train_data[col].nunique()),
                    'zeros_count': int((self.train_data[col] == 0).sum())
                }

        return stats

    def analyze_categorical_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze categorical features in the dataset.

        Returns:
            Dictionary with categorical feature statistics
        """
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()

        stats = {}
        for col in categorical_cols:
            value_counts = self.train_data[col].value_counts()
            stats[col] = {
                'unique_values': int(self.train_data[col].nunique()),
                'missing_count': int(self.train_data[col].isnull().sum()),
                'mode': str(self.train_data[col].mode().iloc[0]) if not self.train_data[col].empty else None,
                'top_5_values': value_counts.head(5).to_dict(),
                'value_counts_dict': value_counts.to_dict()
            }

        return stats

    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Perform correlation analysis on numerical features.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix
        """
        numerical_data = self.train_data.select_dtypes(include=[np.number])
        return numerical_data.corr(method=method)

    def target_correlation(self, top_n: int = 20) -> pd.DataFrame:
        """
        Find features most correlated with target variable.

        Args:
            top_n: Number of top correlations to return

        Returns:
            DataFrame with correlations sorted by absolute value
        """
        if self.target_col not in self.train_data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in training data")

        corr_matrix = self.correlation_analysis()
        target_corr = corr_matrix[self.target_col].drop(self.target_col)

        # Create result DataFrame
        result = pd.DataFrame({
            'feature': target_corr.index,
            'correlation': target_corr.values,
            'abs_correlation': np.abs(target_corr.values)
        }).sort_values('abs_correlation', ascending=False).head(top_n)

        return result.reset_index(drop=True)

    def analyze_target_variable(self) -> Dict[str, Any]:
        """
        Analyze the target variable (SalePrice).

        Returns:
            Dictionary with target variable statistics and insights
        """
        if self.target_col not in self.train_data.columns:
            return {'error': f"Target column '{self.target_col}' not found"}

        target = self.train_data[self.target_col]

        # Basic statistics
        stats = {
            'count': int(len(target)),
            'mean': float(target.mean()),
            'median': float(target.median()),
            'std': float(target.std()),
            'min': float(target.min()),
            'max': float(target.max()),
            'q25': float(target.quantile(0.25)),
            'q75': float(target.quantile(0.75)),
            'skewness': float(target.skew()),
            'kurtosis': float(target.kurtosis())
        }

        # Price ranges
        stats['price_ranges'] = {
            'under_100k': int((target < 100000).sum()),
            '100k_200k': int(((target >= 100000) & (target < 200000)).sum()),
            '200k_300k': int(((target >= 200000) & (target < 300000)).sum()),
            '300k_500k': int(((target >= 300000) & (target < 500000)).sum()),
            'over_500k': int((target >= 500000).sum())
        }

        # Outlier analysis (using IQR method)
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        stats['outliers'] = {
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': int(((target < lower_bound) | (target > upper_bound)).sum()),
            'outlier_percentage': float(((target < lower_bound) | (target > upper_bound)).mean() * 100)
        }

        return stats

    def feature_completeness_analysis(self) -> Dict[str, Any]:
        """
        Analyze feature completeness across the dataset.

        Returns:
            Dictionary with completeness analysis
        """
        total_values = len(self.train_data) * len(self.train_data.columns)
        missing_values = self.train_data.isnull().sum().sum()
        completeness_ratio = (total_values - missing_values) / total_values

        # Categorize columns by missing percentage
        missing_analysis = self.analyze_missing_values()
        missing_pct = missing_analysis['Train_Missing_Percentage']

        categories = {
            'complete': (missing_pct == 0).sum(),
            'low_missing': ((missing_pct > 0) & (missing_pct <= 5)).sum(),
            'medium_missing': ((missing_pct > 5) & (missing_pct <= 25)).sum(),
            'high_missing': ((missing_pct > 25) & (missing_pct <= 50)).sum(),
            'very_high_missing': (missing_pct > 50).sum()
        }

        return {
            'overall_completeness': float(completeness_ratio),
            'total_missing_values': int(missing_values),
            'column_categories': categories,
            'columns_by_category': {
                'complete': missing_analysis[missing_pct == 0].index.tolist(),
                'low_missing': missing_analysis[(missing_pct > 0) & (missing_pct <= 5)].index.tolist(),
                'medium_missing': missing_analysis[(missing_pct > 5) & (missing_pct <= 25)].index.tolist(),
                'high_missing': missing_analysis[(missing_pct > 25) & (missing_pct <= 50)].index.tolist(),
                'very_high_missing': missing_analysis[missing_pct > 50].index.tolist()
            }
        }

    def data_distribution_analysis(self) -> Dict[str, Any]:
        """
        Analyze data distributions for numerical features.

        Returns:
            Dictionary with distribution analysis
        """
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)

        distribution_analysis = {}

        for col in numerical_cols:
            if col in self.train_data.columns:
                data = self.train_data[col].dropna()

                if len(data) > 0:
                    # Normality tests (simple checks)
                    skewness = data.skew()
                    kurtosis = data.kurtosis()

                    # Distribution characteristics
                    distribution_analysis[col] = {
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis),
                        'is_normal_skew': abs(skewness) < 0.5,  # Rule of thumb
                        'is_normal_kurtosis': abs(kurtosis) < 3,  # Rule of thumb
                        'distribution_type': self._classify_distribution(skewness, kurtosis),
                        'zeros_percentage': float((data == 0).mean() * 100),
                        'unique_values': int(data.nunique()),
                        'range': float(data.max() - data.min())
                    }

        return distribution_analysis

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """
        Classify distribution type based on skewness and kurtosis.

        Args:
            skewness: Skewness value
            kurtosis: Kurtosis value

        Returns:
            Distribution classification string
        """
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return "approximately_normal"
        elif skewness > 1:
            return "highly_right_skewed"
        elif skewness > 0.5:
            return "moderately_right_skewed"
        elif skewness < -1:
            return "highly_left_skewed"
        elif skewness < -0.5:
            return "moderately_left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "other"

    def generate_eda_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report.

        Returns:
            Dictionary with complete EDA analysis
        """
        print("Generating comprehensive EDA report...")

        report = {
            'metadata': {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'train_shape': self.original_train_shape,
                'test_shape': self.original_test_shape,
                'total_features': self.train_data.shape[1] - (1 if self.target_col in self.train_data.columns else 0)
            }
        }

        try:
            report['dataset_info'] = {
                'train_shape': self.train_data.shape,
                'test_shape': self.test_data.shape if self.test_data is not None else None,
                'numerical_features_count': len(self.train_data.select_dtypes(include=[np.number]).columns),
                'categorical_features_count': len(self.train_data.select_dtypes(include=['object']).columns),
                'total_features': self.train_data.shape[1] - (1 if self.target_col in self.train_data.columns else 0)
            }

            print("Analyzing missing values...")
            report['missing_values'] = self.analyze_missing_values()

            print("Analyzing numerical features...")
            report['numerical_features'] = self.analyze_numerical_features()

            print("Analyzing categorical features...")
            report['categorical_features'] = self.analyze_categorical_features()

            if self.target_col in self.train_data.columns:
                print("Analyzing target variable...")
                report['target_analysis'] = self.analyze_target_variable()

                print("Calculating feature correlations...")
                report['target_correlations'] = self.target_correlation()

            print("Analyzing feature completeness...")
            report['completeness_analysis'] = self.feature_completeness_analysis()

            print("Analyzing data distributions...")
            report['distribution_analysis'] = self.data_distribution_analysis()

            print("✅ EDA report generation completed successfully")

        except Exception as e:
            print(f"❌ Error during EDA report generation: {e}")
            report['error'] = str(e)

        return report

    def plot_missing_values(self, figsize: Tuple[int, int] = (12, 8), top_n: int = 20) -> None:
        """
        Plot missing values analysis.

        Args:
            figsize: Figure size
            top_n: Number of top missing columns to show
        """
        missing_data = self.analyze_missing_values()
        missing_data = missing_data[missing_data['Train_Missing_Count'] > 0].head(top_n)

        if len(missing_data) == 0:
            print("No missing values found in the dataset")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Missing count
        missing_data['Train_Missing_Count'].plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_title('Missing Values Count (Top Features)')
        axes[0].set_xlabel('Missing Count')

        # Missing percentage
        missing_data['Train_Missing_Percentage'].plot(kind='barh', ax=axes[1], color='lightcoral')
        axes[1].set_title('Missing Values Percentage (Top Features)')
        axes[1].set_xlabel('Missing Percentage (%)')

        plt.tight_layout()
        plt.show()

    def plot_target_distribution(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot target variable distribution analysis.

        Args:
            figsize: Figure size
        """
        if self.target_col not in self.train_data.columns:
            print(f"Target column '{self.target_col}' not found")
            return

        target = self.train_data[self.target_col]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original distribution
        axes[0].hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('SalePrice Distribution')
        axes[0].set_xlabel('Sale Price ($)')
        axes[0].set_ylabel('Frequency')

        # Log distribution
        log_target = np.log(target)
        axes[1].hist(log_target, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].set_title('Log(SalePrice) Distribution')
        axes[1].set_xlabel('Log(Sale Price)')
        axes[1].set_ylabel('Frequency')

        # Box plot
        axes[2].boxplot(target)
        axes[2].set_title('SalePrice Box Plot')
        axes[2].set_ylabel('Sale Price ($)')

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        target_stats = self.analyze_target_variable()
        print(f"\nSalePrice Summary Statistics:")
        print(f"Mean: ${target_stats['mean']:,.2f}")
        print(f"Median: ${target_stats['median']:,.2f}")
        print(f"Std: ${target_stats['std']:,.2f}")
        print(f"Skewness: {target_stats['skewness']:.3f}")
        print(f"Outliers: {target_stats['outliers']['outlier_count']} ({target_stats['outliers']['outlier_percentage']:.1f}%)")

    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10), top_n: int = 20) -> None:
        """
        Plot correlation heatmap for top correlated features.

        Args:
            figsize: Figure size
            top_n: Number of top features to include
        """
        if self.target_col not in self.train_data.columns:
            print(f"Target column '{self.target_col}' not found")
            return

        # Get top correlated features
        top_corr = self.target_correlation(top_n)
        top_features = top_corr['feature'].tolist() + [self.target_col]

        # Create correlation matrix for top features
        corr_matrix = self.train_data[top_features].corr()

        # Plot heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.1,
            cbar_kws={"shrink": 0.8},
            fmt='.2f'
        )
        plt.title(f'Correlation Heatmap - Top {top_n} Features vs SalePrice')
        plt.tight_layout()
        plt.show()

    def print_summary(self) -> None:
        """Print a concise summary of the EDA analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS SUMMARY")
        print("="*60)

        # Basic info
        print(f"\nDATASET OVERVIEW:")
        print(f"Training data shape: {self.train_data.shape}")
        if self.test_data is not None:
            print(f"Test data shape: {self.test_data.shape}")

        # Missing values summary
        missing_summary = self.feature_completeness_analysis()
        print(f"\nDATA COMPLETENESS:")
        print(f"Overall completeness: {missing_summary['overall_completeness']:.1%}")
        print(f"Complete columns: {missing_summary['column_categories']['complete']}")
        print(f"Columns with missing data: {sum(missing_summary['column_categories'].values()) - missing_summary['column_categories']['complete']}")

        # Target analysis
        if self.target_col in self.train_data.columns:
            target_stats = self.analyze_target_variable()
            print(f"\nTARGET VARIABLE (SalePrice):")
            print(f"Mean: ${target_stats['mean']:,.0f}")
            print(f"Median: ${target_stats['median']:,.0f}")
            print(f"Range: ${target_stats['min']:,.0f} - ${target_stats['max']:,.0f}")
            print(f"Skewness: {target_stats['skewness']:.2f}")

            # Top correlations
            top_corr = self.target_correlation(5)
            print(f"\nTOP 5 CORRELATED FEATURES:")
            for _, row in top_corr.iterrows():
                print(f"  {row['feature']}: {row['correlation']:.3f}")

        print("\n" + "="*60)


# Utility functions for EDA
def compare_train_test_distributions(train_data: pd.DataFrame, test_data: pd.DataFrame,
                                   feature: str, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Compare distributions of a feature between train and test sets.

    Args:
        train_data: Training dataset
        test_data: Test dataset
        feature: Feature to compare
        figsize: Figure size
    """
    if feature not in train_data.columns or feature not in test_data.columns:
        print(f"Feature '{feature}' not found in both datasets")
        return

    plt.figure(figsize=figsize)

    if train_data[feature].dtype in ['object', 'category']:
        # Categorical feature
        train_counts = train_data[feature].value_counts(normalize=True)
        test_counts = test_data[feature].value_counts(normalize=True)

        # Align indices
        all_categories = set(train_counts.index) | set(test_counts.index)
        train_aligned = train_counts.reindex(all_categories, fill_value=0)
        test_aligned = test_counts.reindex(all_categories, fill_value=0)

        x = np.arange(len(all_categories))
        width = 0.35

        plt.bar(x - width/2, train_aligned.values, width, label='Train', alpha=0.7)
        plt.bar(x + width/2, test_aligned.values, width, label='Test', alpha=0.7)
        plt.xticks(x, all_categories, rotation=45)
        plt.ylabel('Proportion')

    else:
        # Numerical feature
        plt.hist(train_data[feature].dropna(), alpha=0.7, label='Train', bins=30, density=True)
        plt.hist(test_data[feature].dropna(), alpha=0.7, label='Test', bins=30, density=True)
        plt.xlabel(feature)
        plt.ylabel('Density')

    plt.title(f'Distribution Comparison: {feature}')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Test the module
if __name__ == "__main__":
    print("Testing EDA module...")

    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Id': range(1, 101),
        'SalePrice': np.random.normal(200000, 50000, 100),
        'GrLivArea': np.random.normal(1500, 300, 100),
        'YearBuilt': np.random.randint(1950, 2020, 100),
        'OverallQual': np.random.randint(1, 11, 100),
        'Neighborhood': np.random.choice(['A', 'B', 'C'], 100),
        'MissingFeature': [None if i % 4 == 0 else np.random.normal(100, 20) for i in range(100)]
    })

    try:
        analyzer = EDAAnalyzer(sample_data)
        report = analyzer.generate_eda_report()
        analyzer.print_summary()
        print("\n✅ EDA module working correctly")
    except Exception as e:
        print(f"\n❌ Error in EDA module: {e}")