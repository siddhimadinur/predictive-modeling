# Phase 2: Data Acquisition & Exploration

## Overview
Download the Kaggle dataset, perform comprehensive exploratory data analysis (EDA), and understand the data structure, quality, and relationships for the house price prediction project.

## Objectives
- Download and load the Kaggle House Prices dataset
- Perform comprehensive exploratory data analysis
- Understand feature distributions and relationships
- Identify data quality issues
- Generate insights for feature engineering
- Create visualizations and summary reports

## Step-by-Step Implementation

### 2.1 Data Acquisition

#### 2.1.1 Download Kaggle Dataset
```bash
# Note: You'll need to manually download from Kaggle or set up Kaggle API
# Manual download from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
# Save train.csv and test.csv to data/raw/
```

**Alternative: Create sample data loader for testing**
Create `src/data_loader.py`:
```python
"""
Data loading utilities for house price prediction project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from config.settings import RAW_DATA_DIR, TRAIN_FILE, TEST_FILE
from src.utils import load_data, print_data_info


def load_kaggle_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Kaggle house prices dataset.

    Returns:
        Tuple of (train_data, test_data)
    """
    train_path = RAW_DATA_DIR / TRAIN_FILE
    test_path = RAW_DATA_DIR / TEST_FILE

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found. Please download from Kaggle and place in {RAW_DATA_DIR}"
        )

    train_data = load_data(train_path)
    test_data = load_data(test_path)

    print_data_info(train_data, "Training Data")
    print_data_info(test_data, "Test Data")

    return train_data, test_data


def validate_dataset(train_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
    """
    Validate that the dataset has expected structure.

    Args:
        train_data: Training dataset
        test_data: Test dataset

    Returns:
        True if validation passes
    """
    # Check if target column exists in training data
    if 'SalePrice' not in train_data.columns:
        print("❌ SalePrice column not found in training data")
        return False

    # Check if test data doesn't have target column
    if 'SalePrice' in test_data.columns:
        print("❌ SalePrice column found in test data (should not be present)")
        return False

    # Check if Id column exists in both
    if 'Id' not in train_data.columns or 'Id' not in test_data.columns:
        print("❌ Id column not found in dataset")
        return False

    # Check basic shape expectations
    if train_data.shape[0] < 1000 or test_data.shape[0] < 1000:
        print(f"❌ Unexpected dataset size: train={train_data.shape}, test={test_data.shape}")
        return False

    print("✅ Dataset validation passed")
    return True
```

**Test**: Load and validate data
```python
python -c "
from src.data_loader import load_kaggle_data, validate_dataset
try:
    train, test = load_kaggle_data()
    validate_dataset(train, test)
    print('✓ Data loading successful')
except FileNotFoundError as e:
    print(f'⚠️ {e}')
    print('Please download dataset from Kaggle first')
"
```

### 2.2 Initial Data Exploration

#### 2.2.1 Create EDA Notebook
Create `notebooks/01_exploratory_data_analysis.ipynb`:

```python
# Cell 1: Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append('..')

from src.data_loader import load_kaggle_data, validate_dataset
from src.utils import print_data_info
from config.settings import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("EDA Environment Setup Complete")
```

```python
# Cell 2: Load Data
try:
    train_data, test_data = load_kaggle_data()
    validate_dataset(train_data, test_data)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Total features: {train_data.shape[1] - 1}")  # Exclude target

except FileNotFoundError:
    print("Please download the Kaggle dataset first")
    # For development, create sample data
    print("Creating sample data for development...")
    train_data = pd.DataFrame({
        'Id': range(1, 1001),
        'SalePrice': np.random.normal(200000, 50000, 1000),
        'GrLivArea': np.random.normal(1500, 500, 1000),
        'YearBuilt': np.random.randint(1950, 2010, 1000),
        'OverallQual': np.random.randint(1, 11, 1000)
    })
    test_data = train_data.drop('SalePrice', axis=1).copy()
    print("Sample data created for development")
```

```python
# Cell 3: Basic Data Overview
def basic_data_overview(data, name):
    """Generate basic overview of the dataset."""
    print(f"\n{'='*50}")
    print(f"{name} OVERVIEW")
    print(f"{'='*50}")

    print(f"Shape: {data.shape}")
    print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicate Rows: {data.duplicated().sum()}")

    print(f"\nData Types:")
    print(data.dtypes.value_counts())

    print(f"\nMissing Values:")
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    }).sort_values('Missing_Count', ascending=False)

    print(missing_df[missing_df['Missing_Count'] > 0].head(10))

    return missing_df

# Run basic overview
train_overview = basic_data_overview(train_data, "TRAINING DATA")
test_overview = basic_data_overview(test_data, "TEST DATA")
```

```python
# Cell 4: Target Variable Analysis
if 'SalePrice' in train_data.columns:
    target = train_data['SalePrice']

    # Create subplot for target analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Target Variable (SalePrice) Analysis', fontsize=16)

    # Distribution
    axes[0,0].hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('SalePrice Distribution')
    axes[0,0].set_xlabel('Sale Price ($)')
    axes[0,0].set_ylabel('Frequency')

    # Log distribution
    log_target = np.log(target)
    axes[0,1].hist(log_target, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title('Log(SalePrice) Distribution')
    axes[0,1].set_xlabel('Log(Sale Price)')
    axes[0,1].set_ylabel('Frequency')

    # Box plot
    axes[1,0].boxplot(target)
    axes[1,0].set_title('SalePrice Box Plot')
    axes[1,0].set_ylabel('Sale Price ($)')

    # Q-Q plot
    from scipy import stats
    stats.probplot(target, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot: SalePrice vs Normal Distribution')

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print(f"\nSalePrice Summary Statistics:")
    print(f"Mean: ${target.mean():,.2f}")
    print(f"Median: ${target.median():,.2f}")
    print(f"Std: ${target.std():,.2f}")
    print(f"Min: ${target.min():,.2f}")
    print(f"Max: ${target.max():,.2f}")
    print(f"Skewness: {target.skew():.3f}")
    print(f"Kurtosis: {target.kurtosis():.3f}")
```

#### 2.2.2 Create Data Analysis Module
Create `src/eda.py`:
```python
"""
Exploratory Data Analysis utilities.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from config.settings import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


class EDAAnalyzer:
    """Comprehensive EDA analysis class."""

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

    def analyze_numerical_features(self) -> Dict:
        """
        Analyze numerical features.

        Returns:
            Dictionary with numerical feature statistics
        """
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)

        stats = {}
        for col in numerical_cols:
            if col in self.train_data.columns:
                stats[col] = {
                    'mean': self.train_data[col].mean(),
                    'median': self.train_data[col].median(),
                    'std': self.train_data[col].std(),
                    'min': self.train_data[col].min(),
                    'max': self.train_data[col].max(),
                    'skewness': self.train_data[col].skew(),
                    'kurtosis': self.train_data[col].kurtosis(),
                    'missing_count': self.train_data[col].isnull().sum(),
                    'unique_values': self.train_data[col].nunique()
                }

        return stats

    def analyze_categorical_features(self) -> Dict:
        """
        Analyze categorical features.

        Returns:
            Dictionary with categorical feature statistics
        """
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()

        stats = {}
        for col in categorical_cols:
            stats[col] = {
                'unique_values': self.train_data[col].nunique(),
                'missing_count': self.train_data[col].isnull().sum(),
                'mode': self.train_data[col].mode().iloc[0] if not self.train_data[col].empty else None,
                'value_counts': self.train_data[col].value_counts().head(10).to_dict()
            }

        return stats

    def correlation_analysis(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Perform correlation analysis.

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
            raise ValueError(f"Target column '{self.target_col}' not found")

        corr_matrix = self.correlation_analysis()
        target_corr = corr_matrix[self.target_col].drop(self.target_col)

        return target_corr.abs().sort_values(ascending=False).head(top_n).to_frame('correlation')

    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot correlation heatmap.

        Args:
            figsize: Figure size tuple
        """
        corr_matrix = self.correlation_analysis()

        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.1
        )
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def generate_eda_report(self) -> Dict:
        """
        Generate comprehensive EDA report.

        Returns:
            Dictionary with complete EDA analysis
        """
        report = {
            'dataset_info': {
                'train_shape': self.train_data.shape,
                'test_shape': self.test_data.shape if self.test_data is not None else None,
                'total_features': self.train_data.shape[1] - (1 if self.target_col in self.train_data.columns else 0)
            },
            'missing_values': self.analyze_missing_values(),
            'numerical_features': self.analyze_numerical_features(),
            'categorical_features': self.analyze_categorical_features(),
            'target_correlations': self.target_correlation() if self.target_col in self.train_data.columns else None
        }

        return report
```

**Test**: Run EDA analysis
```python
python -c "
from src.data_loader import load_kaggle_data
from src.eda import EDAAnalyzer

try:
    train, test = load_kaggle_data()
    analyzer = EDAAnalyzer(train, test)
    report = analyzer.generate_eda_report()
    print('✓ EDA analysis completed successfully')
    print(f'Dataset shape: {report[\"dataset_info\"][\"train_shape\"]}')
except Exception as e:
    print(f'Note: {e}')
    print('EDA module created successfully - requires dataset for full testing')
"
```

### 2.3 Advanced Exploratory Analysis

#### 2.3.1 Create Visualization Module
Create `src/visualization.py`:
```python
"""
Visualization utilities for house price prediction project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple


class HousePriceVisualizer:
    """Visualization class for house price data."""

    def __init__(self, data: pd.DataFrame, target_col: str = 'SalePrice'):
        """
        Initialize visualizer.

        Args:
            data: Dataset to visualize
            target_col: Target column name
        """
        self.data = data.copy()
        self.target_col = target_col

    def plot_target_distribution(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot target variable distribution."""
        if self.target_col not in self.data.columns:
            print(f"Target column '{self.target_col}' not found")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        target = self.data[self.target_col]

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

    def plot_feature_vs_target(self, feature: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot feature relationship with target.

        Args:
            feature: Feature column name
            figsize: Figure size
        """
        if feature not in self.data.columns or self.target_col not in self.data.columns:
            print(f"Column not found: {feature} or {self.target_col}")
            return

        plt.figure(figsize=figsize)

        if self.data[feature].dtype in ['object', 'category']:
            # Categorical feature
            sns.boxplot(data=self.data, x=feature, y=self.target_col)
            plt.xticks(rotation=45)
        else:
            # Numerical feature
            plt.scatter(self.data[feature], self.data[self.target_col], alpha=0.6)
            plt.xlabel(feature)
            plt.ylabel(self.target_col)

            # Add correlation coefficient
            corr = self.data[feature].corr(self.data[self.target_col])
            plt.title(f'{feature} vs {self.target_col} (correlation: {corr:.3f})')

        plt.tight_layout()
        plt.show()

    def plot_missing_values(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot missing values analysis."""
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100

        # Only show columns with missing values
        missing_data = missing[missing > 0].sort_values(ascending=True)
        missing_pct_data = missing_pct[missing_pct > 0].sort_values(ascending=True)

        if len(missing_data) == 0:
            print("No missing values found in the dataset")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Missing count
        missing_data.plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_title('Missing Values Count')
        axes[0].set_xlabel('Count')

        # Missing percentage
        missing_pct_data.plot(kind='barh', ax=axes[1], color='lightcoral')
        axes[1].set_title('Missing Values Percentage')
        axes[1].set_xlabel('Percentage (%)')

        plt.tight_layout()
        plt.show()

    def plot_top_correlations(self, top_n: int = 15, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot top correlations with target variable.

        Args:
            top_n: Number of top correlations to show
            figsize: Figure size
        """
        if self.target_col not in self.data.columns:
            print(f"Target column '{self.target_col}' not found")
            return

        # Calculate correlations
        numerical_data = self.data.select_dtypes(include=[np.number])
        correlations = numerical_data.corr()[self.target_col].drop(self.target_col)
        top_corr = correlations.abs().sort_values(ascending=True).tail(top_n)

        # Create colors based on positive/negative correlation
        colors = ['red' if x < 0 else 'green' for x in correlations[top_corr.index]]

        plt.figure(figsize=figsize)
        top_corr.plot(kind='barh', color=colors)
        plt.title(f'Top {top_n} Features Correlated with {self.target_col}')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.show()

    def create_interactive_scatter(self, x_feature: str, y_feature: str) -> go.Figure:
        """
        Create interactive scatter plot.

        Args:
            x_feature: X-axis feature
            y_feature: Y-axis feature

        Returns:
            Plotly figure object
        """
        fig = px.scatter(
            self.data,
            x=x_feature,
            y=y_feature,
            color=self.target_col if self.target_col in self.data.columns else None,
            title=f'{x_feature} vs {y_feature}',
            hover_data=['Id'] if 'Id' in self.data.columns else None
        )
        return fig

    def create_correlation_matrix_interactive(self) -> go.Figure:
        """Create interactive correlation matrix."""
        numerical_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numerical_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(
            title='Interactive Correlation Matrix',
            width=800,
            height=800
        )

        return fig
```

#### 2.3.2 Continue EDA Notebook with Advanced Analysis

Add to `notebooks/01_exploratory_data_analysis.ipynb`:

```python
# Cell 5: Missing Values Analysis
from src.visualization import HousePriceVisualizer

visualizer = HousePriceVisualizer(train_data)
visualizer.plot_missing_values()

# Detailed missing values analysis
missing_analysis = analyzer.analyze_missing_values()
print("\nTop 15 columns with missing values:")
print(missing_analysis[missing_analysis['Train_Missing_Count'] > 0].head(15))
```

```python
# Cell 6: Numerical Features Analysis
numerical_stats = analyzer.analyze_numerical_features()

# Plot distributions of key numerical features
key_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'YearBuilt', 'OverallQual']
existing_features = [f for f in key_features if f in train_data.columns]

if existing_features:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, feature in enumerate(existing_features[:6]):
        if i < len(axes):
            train_data[feature].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')

    # Remove empty subplots
    for i in range(len(existing_features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
```

```python
# Cell 7: Categorical Features Analysis
categorical_stats = analyzer.analyze_categorical_features()

# Visualize key categorical features
cat_features = ['Neighborhood', 'OverallQual', 'ExterQual', 'KitchenQual']
existing_cat_features = [f for f in cat_features if f in train_data.columns]

for feature in existing_cat_features[:2]:  # Show first 2 to avoid overcrowding
    if 'SalePrice' in train_data.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=train_data, x=feature, y='SalePrice')
        plt.xticks(rotation=45)
        plt.title(f'SalePrice by {feature}')
        plt.tight_layout()
        plt.show()
```

```python
# Cell 8: Correlation Analysis
# Top correlations with target
if 'SalePrice' in train_data.columns:
    top_correlations = analyzer.target_correlation(top_n=15)
    print("Top 15 features correlated with SalePrice:")
    print(top_correlations)

    # Visualize correlations
    visualizer.plot_top_correlations(top_n=15)
```

### 2.4 Data Quality Assessment

#### 2.4.1 Create Data Quality Module
Create `src/data_quality.py`:
```python
"""
Data quality assessment utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class DataQualityAssessor:
    """Assess data quality issues and provide recommendations."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize data quality assessor.

        Args:
            data: Dataset to assess
        """
        self.data = data.copy()

    def assess_missing_values(self) -> Dict[str, Any]:
        """
        Assess missing value patterns.

        Returns:
            Dictionary with missing value assessment
        """
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100

        # Categorize missing value severity
        high_missing = missing_percentages[missing_percentages > 50].index.tolist()
        medium_missing = missing_percentages[(missing_percentages > 20) & (missing_percentages <= 50)].index.tolist()
        low_missing = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 20)].index.tolist()

        return {
            'total_missing_values': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'high_missing_columns': high_missing,
            'medium_missing_columns': medium_missing,
            'low_missing_columns': low_missing,
            'recommendations': self._get_missing_value_recommendations(high_missing, medium_missing, low_missing)
        }

    def assess_data_types(self) -> Dict[str, Any]:
        """
        Assess data type appropriateness.

        Returns:
            Dictionary with data type assessment
        """
        type_issues = []

        for col in self.data.columns:
            # Check for potential numeric columns stored as objects
            if self.data[col].dtype == 'object':
                # Try to convert to numeric
                numeric_converted = pd.to_numeric(self.data[col], errors='coerce')
                non_null_original = self.data[col].notna().sum()
                non_null_converted = numeric_converted.notna().sum()

                # If most values can be converted to numeric, flag as potential issue
                if non_null_converted / non_null_original > 0.8:
                    type_issues.append({
                        'column': col,
                        'current_type': 'object',
                        'suggested_type': 'numeric',
                        'conversion_success_rate': non_null_converted / non_null_original
                    })

        return {
            'type_issues': type_issues,
            'data_types_summary': self.data.dtypes.value_counts().to_dict()
        }

    def assess_outliers(self, numerical_columns: List[str] = None) -> Dict[str, Any]:
        """
        Assess outliers in numerical columns.

        Args:
            numerical_columns: List of columns to check (default: all numerical)

        Returns:
            Dictionary with outlier assessment
        """
        if numerical_columns is None:
            numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        outlier_analysis = {}

        for col in numerical_columns:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]

                outlier_analysis[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(self.data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_indices': outliers.index.tolist()
                }

        return outlier_analysis

    def assess_duplicates(self) -> Dict[str, Any]:
        """
        Assess duplicate records.

        Returns:
            Dictionary with duplicate assessment
        """
        # Full duplicates
        full_duplicates = self.data.duplicated().sum()

        # Duplicates excluding ID column (if exists)
        columns_to_check = [col for col in self.data.columns if col.lower() not in ['id', 'index']]
        feature_duplicates = self.data[columns_to_check].duplicated().sum() if columns_to_check else 0

        return {
            'full_duplicates': full_duplicates,
            'feature_duplicates': feature_duplicates,
            'duplicate_indices': self.data[self.data.duplicated()].index.tolist()
        }

    def assess_cardinality(self) -> Dict[str, Any]:
        """
        Assess feature cardinality.

        Returns:
            Dictionary with cardinality assessment
        """
        cardinality_analysis = {}

        for col in self.data.columns:
            unique_count = self.data[col].nunique()
            total_count = len(self.data)
            cardinality_ratio = unique_count / total_count

            # Categorize cardinality
            if cardinality_ratio == 1.0:
                category = 'unique_identifier'
            elif cardinality_ratio > 0.95:
                category = 'high_cardinality'
            elif cardinality_ratio < 0.01:
                category = 'low_cardinality'
            else:
                category = 'normal_cardinality'

            cardinality_analysis[col] = {
                'unique_count': unique_count,
                'total_count': total_count,
                'cardinality_ratio': cardinality_ratio,
                'category': category
            }

        return cardinality_analysis

    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Returns:
            Complete data quality assessment
        """
        return {
            'dataset_overview': {
                'shape': self.data.shape,
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
                'data_types': self.data.dtypes.value_counts().to_dict()
            },
            'missing_values': self.assess_missing_values(),
            'data_types': self.assess_data_types(),
            'outliers': self.assess_outliers(),
            'duplicates': self.assess_duplicates(),
            'cardinality': self.assess_cardinality()
        }

    def _get_missing_value_recommendations(self, high: List[str], medium: List[str], low: List[str]) -> List[str]:
        """Generate recommendations for handling missing values."""
        recommendations = []

        if high:
            recommendations.append(f"Consider dropping columns with >50% missing: {high}")
        if medium:
            recommendations.append(f"Investigate patterns in medium missing columns: {medium}")
        if low:
            recommendations.append(f"Apply appropriate imputation for low missing columns: {low}")

        return recommendations
```

**Test**: Data quality assessment
```python
python -c "
from src.data_quality import DataQualityAssessor
import pandas as pd
import numpy as np

# Create sample data for testing
sample_data = pd.DataFrame({
    'feature1': [1, 2, 3, None, 5],
    'feature2': ['A', 'B', 'C', 'D', 'E'],
    'feature3': [10, 20, 30, 1000, 50]  # Contains outlier
})

assessor = DataQualityAssessor(sample_data)
quality_report = assessor.generate_quality_report()
print('✓ Data quality assessment module working correctly')
print(f'Sample assessment - Missing values: {quality_report[\"missing_values\"][\"total_missing_values\"]}')
"
```

### 2.5 Final EDA Report Generation

#### 2.5.1 Create Report Generator
Create `src/eda_report.py`:
```python
"""
EDA report generation utilities.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.eda import EDAAnalyzer
from src.data_quality import DataQualityAssessor


def generate_eda_report(train_data: pd.DataFrame, test_data: pd.DataFrame = None,
                       output_dir: Path = None) -> Dict[str, Any]:
    """
    Generate comprehensive EDA report.

    Args:
        train_data: Training dataset
        test_data: Test dataset (optional)
        output_dir: Directory to save report

    Returns:
        Complete EDA report dictionary
    """
    if output_dir is None:
        output_dir = Path("reports")

    output_dir.mkdir(exist_ok=True)

    # Initialize analyzers
    eda_analyzer = EDAAnalyzer(train_data, test_data)
    quality_assessor = DataQualityAssessor(train_data)

    # Generate analyses
    eda_report = eda_analyzer.generate_eda_report()
    quality_report = quality_assessor.generate_quality_report()

    # Combine reports
    full_report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'train_dataset_shape': train_data.shape,
            'test_dataset_shape': test_data.shape if test_data is not None else None
        },
        'eda_analysis': eda_report,
        'data_quality': quality_report
    }

    # Save report
    report_file = output_dir / f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj

    # Recursively convert numpy types
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert_numpy_types(obj)

    serializable_report = recursive_convert(full_report)

    with open(report_file, 'w') as f:
        json.dump(serializable_report, f, indent=2, default=str)

    print(f"EDA report saved to: {report_file}")
    return full_report


def create_eda_summary(report: Dict[str, Any]) -> str:
    """
    Create a text summary of the EDA report.

    Args:
        report: EDA report dictionary

    Returns:
        Text summary string
    """
    summary = []
    summary.append("=" * 60)
    summary.append("HOUSE PRICE PREDICTION - EDA SUMMARY")
    summary.append("=" * 60)

    # Dataset overview
    meta = report['metadata']
    summary.append(f"\nDATASET OVERVIEW:")
    summary.append(f"Training data shape: {meta['train_dataset_shape']}")
    if meta['test_dataset_shape']:
        summary.append(f"Test data shape: {meta['test_dataset_shape']}")
    summary.append(f"Report generated: {meta['generated_at']}")

    # Data quality highlights
    quality = report['data_quality']
    summary.append(f"\nDATA QUALITY HIGHLIGHTS:")
    summary.append(f"Total missing values: {quality['missing_values']['total_missing_values']}")
    summary.append(f"Duplicate records: {quality['duplicates']['full_duplicates']}")

    missing_cols = len(quality['missing_values']['columns_with_missing'])
    summary.append(f"Columns with missing data: {missing_cols}")

    # Feature insights
    if 'target_correlations' in report['eda_analysis'] and report['eda_analysis']['target_correlations'] is not None:
        summary.append(f"\nTOP CORRELATED FEATURES:")
        top_corr = report['eda_analysis']['target_correlations']
        for feature, corr in list(top_corr.items())[:5]:
            summary.append(f"  {feature}: {corr:.3f}")

    # Recommendations
    summary.append(f"\nRECOMMENDations:")
    recommendations = quality['missing_values']['recommendations']
    for i, rec in enumerate(recommendations, 1):
        summary.append(f"  {i}. {rec}")

    return "\n".join(summary)
```

### 2.6 Testing and Validation

Create `tests/test_phase2.py`:
```python
"""
Tests for Phase 2: Data Acquisition & Exploration
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('..')

from src.data_loader import validate_dataset
from src.eda import EDAAnalyzer
from src.data_quality import DataQualityAssessor
from src.visualization import HousePriceVisualizer


class TestPhase2(unittest.TestCase):
    """Test Phase 2 functionality."""

    def setUp(self):
        """Set up test data."""
        # Create sample data for testing
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Id': range(1, 101),
            'SalePrice': np.random.normal(200000, 50000, 100),
            'GrLivArea': np.random.normal(1500, 300, 100),
            'YearBuilt': np.random.randint(1950, 2020, 100),
            'OverallQual': np.random.randint(1, 11, 100),
            'Neighborhood': np.random.choice(['A', 'B', 'C'], 100),
            'MissingFeature': [None] * 50 + list(range(50))
        })

        self.test_data = self.sample_data.drop('SalePrice', axis=1)

    def test_data_validation(self):
        """Test data validation function."""
        # Test valid data
        self.assertTrue(validate_dataset(self.sample_data, self.test_data))

        # Test invalid data (no SalePrice in train)
        invalid_train = self.sample_data.drop('SalePrice', axis=1)
        self.assertFalse(validate_dataset(invalid_train, self.test_data))

    def test_eda_analyzer(self):
        """Test EDA analyzer functionality."""
        analyzer = EDAAnalyzer(self.sample_data, self.test_data)

        # Test missing values analysis
        missing_analysis = analyzer.analyze_missing_values()
        self.assertIsInstance(missing_analysis, pd.DataFrame)
        self.assertIn('Train_Missing_Count', missing_analysis.columns)

        # Test numerical features analysis
        numerical_stats = analyzer.analyze_numerical_features()
        self.assertIsInstance(numerical_stats, dict)

        # Test correlation analysis
        correlations = analyzer.correlation_analysis()
        self.assertIsInstance(correlations, pd.DataFrame)

        # Test target correlation
        target_corr = analyzer.target_correlation(top_n=5)
        self.assertIsInstance(target_corr, pd.DataFrame)
        self.assertEqual(len(target_corr), 5)

    def test_data_quality_assessor(self):
        """Test data quality assessor."""
        assessor = DataQualityAssessor(self.sample_data)

        # Test missing values assessment
        missing_assessment = assessor.assess_missing_values()
        self.assertIsInstance(missing_assessment, dict)
        self.assertIn('total_missing_values', missing_assessment)

        # Test outliers assessment
        outlier_assessment = assessor.assess_outliers()
        self.assertIsInstance(outlier_assessment, dict)

        # Test full quality report
        quality_report = assessor.generate_quality_report()
        self.assertIsInstance(quality_report, dict)
        self.assertIn('dataset_overview', quality_report)

    def test_visualizer(self):
        """Test visualizer initialization."""
        visualizer = HousePriceVisualizer(self.sample_data)
        self.assertEqual(visualizer.target_col, 'SalePrice')
        self.assertEqual(len(visualizer.data), 100)


if __name__ == '__main__':
    unittest.main()
```

**Test**: Run Phase 2 tests
```bash
cd tests
python test_phase2.py
```

## Deliverables
- [ ] Kaggle dataset downloaded and loaded
- [ ] Comprehensive EDA notebook with visualizations
- [ ] Data analysis modules (EDA, visualization, data quality)
- [ ] Missing value analysis and recommendations
- [ ] Feature correlation analysis
- [ ] Data quality assessment report
- [ ] Automated testing suite
- [ ] EDA summary report generation

## Success Criteria
- Dataset successfully loaded and validated
- EDA analysis identifies key patterns and relationships
- Data quality issues documented with recommendations
- Visualization modules create informative plots
- All tests pass successfully
- Comprehensive EDA report generated

## Next Phase
Proceed to **Phase 3: Data Processing & Feature Engineering** with insights from EDA analysis.