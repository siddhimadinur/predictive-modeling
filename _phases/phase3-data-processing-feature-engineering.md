# Phase 3: Data Processing & Feature Engineering

## Overview
Clean the dataset, handle missing values, engineer new features, and prepare data for machine learning models based on insights from Phase 2 EDA analysis.

## Objectives
- Clean and preprocess raw data
- Handle missing values systematically
- Engineer meaningful features
- Scale and transform features appropriately
- Create feature pipelines for reproducible preprocessing
- Split data into train/validation/test sets
- Save processed datasets for modeling

## Step-by-Step Implementation

### 3.1 Data Cleaning Foundation

#### 3.1.1 Create Data Cleaning Module
Create `src/data_cleaning.py`:
```python
"""
Data cleaning utilities for house price prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from config.settings import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils import save_data, print_data_info


class DataCleaner:
    """Comprehensive data cleaning class."""

    def __init__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None):
        """
        Initialize data cleaner.

        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
        """
        self.train_data = train_data.copy()
        self.test_data = test_data.copy() if test_data is not None else None
        self.cleaning_log = []

    def log_operation(self, operation: str, details: str) -> None:
        """
        Log cleaning operations.

        Args:
            operation: Type of operation
            details: Operation details
        """
        self.cleaning_log.append({
            'operation': operation,
            'details': details,
            'train_shape_after': self.train_data.shape,
            'test_shape_after': self.test_data.shape if self.test_data is not None else None
        })

    def remove_duplicates(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove duplicate records.

        Returns:
            Tuple of cleaned (train_data, test_data)
        """
        # Remove full duplicates
        initial_train_count = len(self.train_data)
        self.train_data = self.train_data.drop_duplicates()
        train_duplicates_removed = initial_train_count - len(self.train_data)

        if self.test_data is not None:
            initial_test_count = len(self.test_data)
            self.test_data = self.test_data.drop_duplicates()
            test_duplicates_removed = initial_test_count - len(self.test_data)
        else:
            test_duplicates_removed = 0

        self.log_operation(
            'remove_duplicates',
            f'Removed {train_duplicates_removed} train duplicates, {test_duplicates_removed} test duplicates'
        )

        return self.train_data, self.test_data

    def handle_outliers(self, columns: List[str] = None, method: str = 'iqr',
                       factor: float = 1.5) -> Tuple[pd.DataFrame, List[int]]:
        """
        Handle outliers in numerical columns.

        Args:
            columns: Columns to process (default: all numerical)
            method: Outlier detection method ('iqr', 'zscore')
            factor: Outlier threshold factor

        Returns:
            Tuple of (cleaned_data, outlier_indices)
        """
        if columns is None:
            columns = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target and ID columns
            columns = [col for col in columns if col not in ['SalePrice', 'Id']]

        outlier_indices = []

        for col in columns:
            if col in self.train_data.columns:
                if method == 'iqr':
                    Q1 = self.train_data[col].quantile(0.25)
                    Q3 = self.train_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR

                    outliers = self.train_data[
                        (self.train_data[col] < lower_bound) | (self.train_data[col] > upper_bound)
                    ].index.tolist()

                elif method == 'zscore':
                    z_scores = np.abs((self.train_data[col] - self.train_data[col].mean()) / self.train_data[col].std())
                    outliers = self.train_data[z_scores > factor].index.tolist()

                outlier_indices.extend(outliers)

        # Remove outliers from training data only
        outlier_indices = list(set(outlier_indices))
        initial_count = len(self.train_data)
        self.train_data = self.train_data.drop(outlier_indices)

        self.log_operation(
            'handle_outliers',
            f'Removed {initial_count - len(self.train_data)} outlier records using {method} method'
        )

        return self.train_data, outlier_indices

    def standardize_column_names(self) -> None:
        """Standardize column names."""
        # Convert to snake_case and remove special characters
        def clean_column_name(name):
            # Replace spaces and special characters with underscores
            name = str(name).replace(' ', '_').replace('-', '_')
            # Remove other special characters
            name = ''.join(char if char.isalnum() or char == '_' else '' for char in name)
            # Convert to lowercase
            name = name.lower()
            # Remove consecutive underscores
            while '__' in name:
                name = name.replace('__', '_')
            # Remove leading/trailing underscores
            name = name.strip('_')
            return name

        # Create mapping of old to new names
        column_mapping = {col: clean_column_name(col) for col in self.train_data.columns}

        # Apply to training data
        self.train_data = self.train_data.rename(columns=column_mapping)

        # Apply to test data if exists
        if self.test_data is not None:
            test_mapping = {col: clean_column_name(col) for col in self.test_data.columns}
            self.test_data = self.test_data.rename(columns=test_mapping)

        self.log_operation('standardize_column_names', 'Standardized all column names to snake_case')

    def convert_data_types(self) -> None:
        """Convert data types appropriately."""
        type_conversions = []

        for col in self.train_data.columns:
            original_dtype = self.train_data[col].dtype

            # Try to convert object columns to numeric if possible
            if original_dtype == 'object':
                # Check if it can be converted to numeric
                numeric_converted = pd.to_numeric(self.train_data[col], errors='coerce')
                non_null_original = self.train_data[col].notna().sum()
                non_null_converted = numeric_converted.notna().sum()

                # If most values can be converted (>90%), convert to numeric
                if non_null_converted / non_null_original > 0.9:
                    self.train_data[col] = numeric_converted
                    if self.test_data is not None and col in self.test_data.columns:
                        self.test_data[col] = pd.to_numeric(self.test_data[col], errors='coerce')
                    type_conversions.append(f'{col}: {original_dtype} -> numeric')

        self.log_operation('convert_data_types', f'Converted types: {type_conversions}')

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of all cleaning operations.

        Returns:
            Dictionary with cleaning summary
        """
        return {
            'operations_performed': len(self.cleaning_log),
            'cleaning_log': self.cleaning_log,
            'final_train_shape': self.train_data.shape,
            'final_test_shape': self.test_data.shape if self.test_data is not None else None
        }
```

**Test**: Basic data cleaning
```python
python -c "
from src.data_cleaning import DataCleaner
import pandas as pd
import numpy as np

# Create sample data with issues
sample_data = pd.DataFrame({
    'Feature 1': [1, 2, 3, 3, 1000],  # Contains duplicate and outlier
    'Feature_2': ['1', '2', '3', 'invalid', '5'],  # Mixed types
    'SalePrice': [100000, 200000, 300000, 300000, 250000]
})

cleaner = DataCleaner(sample_data)
cleaned_train, _ = cleaner.remove_duplicates()
print('✓ Data cleaning module working correctly')
print(f'Cleaning operations: {len(cleaner.cleaning_log)}')
"
```

### 3.2 Missing Value Handling

#### 3.2.1 Create Missing Value Handler
Create `src/missing_value_handler.py`:
```python
"""
Missing value handling strategies for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class MissingValueHandler:
    """Handle missing values with various strategies."""

    def __init__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None):
        """
        Initialize missing value handler.

        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
        """
        self.train_data = train_data.copy()
        self.test_data = test_data.copy() if test_data is not None else None
        self.imputation_strategies = {}
        self.fitted_imputers = {}

    def analyze_missing_patterns(self) -> Dict[str, Any]:
        """
        Analyze missing value patterns.

        Returns:
            Dictionary with missing pattern analysis
        """
        missing_analysis = {}

        # Get missing value counts and percentages
        train_missing = self.train_data.isnull().sum()
        train_missing_pct = (train_missing / len(self.train_data)) * 100

        analysis = {
            'train_missing_counts': train_missing[train_missing > 0].to_dict(),
            'train_missing_percentages': train_missing_pct[train_missing_pct > 0].to_dict()
        }

        if self.test_data is not None:
            test_missing = self.test_data.isnull().sum()
            test_missing_pct = (test_missing / len(self.test_data)) * 100
            analysis['test_missing_counts'] = test_missing[test_missing > 0].to_dict()
            analysis['test_missing_percentages'] = test_missing_pct[test_missing_pct > 0].to_dict()

        # Categorize missing severity
        high_missing = train_missing_pct[train_missing_pct > 50].index.tolist()
        medium_missing = train_missing_pct[(train_missing_pct > 20) & (train_missing_pct <= 50)].index.tolist()
        low_missing = train_missing_pct[(train_missing_pct > 0) & (train_missing_pct <= 20)].index.tolist()

        analysis.update({
            'high_missing_columns': high_missing,
            'medium_missing_columns': medium_missing,
            'low_missing_columns': low_missing
        })

        return analysis

    def create_imputation_strategy(self, missing_analysis: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Create imputation strategy based on missing value analysis.

        Args:
            missing_analysis: Missing value analysis (if None, will compute)

        Returns:
            Dictionary mapping columns to imputation strategies
        """
        if missing_analysis is None:
            missing_analysis = self.analyze_missing_patterns()

        strategies = {}

        # High missing columns (>50%) - consider dropping or special handling
        for col in missing_analysis['high_missing_columns']:
            if col in self.train_data.columns:
                # Check if it's a categorical or numerical column
                if self.train_data[col].dtype == 'object':
                    strategies[col] = 'fill_missing_category'
                else:
                    strategies[col] = 'drop_column'  # Consider dropping high missing numerical

        # Medium missing columns (20-50%) - use advanced imputation
        for col in missing_analysis['medium_missing_columns']:
            if col in self.train_data.columns:
                if self.train_data[col].dtype == 'object':
                    strategies[col] = 'mode_imputation'
                else:
                    strategies[col] = 'knn_imputation'

        # Low missing columns (0-20%) - use simple imputation
        for col in missing_analysis['low_missing_columns']:
            if col in self.train_data.columns:
                if self.train_data[col].dtype == 'object':
                    strategies[col] = 'mode_imputation'
                else:
                    strategies[col] = 'median_imputation'

        self.imputation_strategies = strategies
        return strategies

    def apply_imputation(self, strategy_override: Dict[str, str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply imputation strategies.

        Args:
            strategy_override: Override default strategies

        Returns:
            Tuple of imputed (train_data, test_data)
        """
        if strategy_override:
            strategies = strategy_override
        else:
            strategies = self.imputation_strategies

        if not strategies:
            # Create default strategies
            strategies = self.create_imputation_strategy()

        train_imputed = self.train_data.copy()
        test_imputed = self.test_data.copy() if self.test_data is not None else None

        for column, strategy in strategies.items():
            if column not in train_imputed.columns:
                continue

            if strategy == 'drop_column':
                train_imputed = train_imputed.drop(column, axis=1)
                if test_imputed is not None and column in test_imputed.columns:
                    test_imputed = test_imputed.drop(column, axis=1)

            elif strategy == 'median_imputation':
                imputer = SimpleImputer(strategy='median')
                train_imputed[column] = imputer.fit_transform(train_imputed[[column]]).ravel()
                self.fitted_imputers[column] = imputer

                if test_imputed is not None and column in test_imputed.columns:
                    test_imputed[column] = imputer.transform(test_imputed[[column]]).ravel()

            elif strategy == 'mean_imputation':
                imputer = SimpleImputer(strategy='mean')
                train_imputed[column] = imputer.fit_transform(train_imputed[[column]]).ravel()
                self.fitted_imputers[column] = imputer

                if test_imputed is not None and column in test_imputed.columns:
                    test_imputed[column] = imputer.transform(test_imputed[[column]]).ravel()

            elif strategy == 'mode_imputation':
                imputer = SimpleImputer(strategy='most_frequent')
                train_imputed[column] = imputer.fit_transform(train_imputed[[column]]).ravel()
                self.fitted_imputers[column] = imputer

                if test_imputed is not None and column in test_imputed.columns:
                    test_imputed[column] = imputer.transform(test_imputed[[column]]).ravel()

            elif strategy == 'fill_missing_category':
                train_imputed[column] = train_imputed[column].fillna('Missing')
                if test_imputed is not None and column in test_imputed.columns:
                    test_imputed[column] = test_imputed[column].fillna('Missing')

            elif strategy == 'knn_imputation':
                # For KNN imputation, we need to select only numerical columns
                numerical_cols = train_imputed.select_dtypes(include=[np.number]).columns.tolist()
                if column in numerical_cols and len(numerical_cols) > 1:
                    imputer = KNNImputer(n_neighbors=5)

                    # Fit on numerical columns only
                    train_numerical = train_imputed[numerical_cols]
                    imputed_values = imputer.fit_transform(train_numerical)

                    # Update only the target column
                    col_index = numerical_cols.index(column)
                    train_imputed[column] = imputed_values[:, col_index]
                    self.fitted_imputers[column] = (imputer, numerical_cols)

                    if test_imputed is not None and column in test_imputed.columns:
                        test_numerical = test_imputed[numerical_cols]
                        test_imputed_values = imputer.transform(test_numerical)
                        test_imputed[column] = test_imputed_values[:, col_index]

        return train_imputed, test_imputed

    def validate_imputation(self, train_imputed: pd.DataFrame,
                          test_imputed: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Validate imputation results.

        Args:
            train_imputed: Imputed training data
            test_imputed: Imputed test data

        Returns:
            Validation results
        """
        validation_results = {
            'train_missing_before': self.train_data.isnull().sum().sum(),
            'train_missing_after': train_imputed.isnull().sum().sum(),
            'train_shape_before': self.train_data.shape,
            'train_shape_after': train_imputed.shape
        }

        if test_imputed is not None and self.test_data is not None:
            validation_results.update({
                'test_missing_before': self.test_data.isnull().sum().sum(),
                'test_missing_after': test_imputed.isnull().sum().sum(),
                'test_shape_before': self.test_data.shape,
                'test_shape_after': test_imputed.shape
            })

        # Check for any remaining missing values
        remaining_missing_train = train_imputed.isnull().sum()
        remaining_missing_train = remaining_missing_train[remaining_missing_train > 0]

        if len(remaining_missing_train) > 0:
            validation_results['remaining_missing_columns'] = remaining_missing_train.to_dict()
        else:
            validation_results['imputation_complete'] = True

        return validation_results

    def get_imputation_summary(self) -> Dict[str, Any]:
        """
        Get summary of imputation strategies and results.

        Returns:
            Imputation summary
        """
        return {
            'strategies_used': self.imputation_strategies,
            'fitted_imputers': list(self.fitted_imputers.keys()),
            'total_strategies': len(self.imputation_strategies)
        }
```

**Test**: Missing value handling
```python
python -c "
from src.missing_value_handler import MissingValueHandler
import pandas as pd
import numpy as np

# Create sample data with missing values
sample_data = pd.DataFrame({
    'numerical_feature': [1, 2, None, 4, 5, None],
    'categorical_feature': ['A', 'B', None, 'A', 'B', None],
    'high_missing': [None, None, None, None, 1, 2],
    'SalePrice': [100, 200, 300, 400, 500, 600]
})

handler = MissingValueHandler(sample_data)
analysis = handler.analyze_missing_patterns()
strategies = handler.create_imputation_strategy(analysis)
train_imputed, _ = handler.apply_imputation()

print('✓ Missing value handler working correctly')
print(f'Strategies created: {len(strategies)}')
print(f'Missing values before: {sample_data.isnull().sum().sum()}')
print(f'Missing values after: {train_imputed.isnull().sum().sum()}')
"
```

### 3.3 Feature Engineering

#### 3.3.1 Create Feature Engineering Module
Create `src/feature_engineering.py`:
```python
"""
Feature engineering utilities for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


class FeatureEngineer:
    """Comprehensive feature engineering class."""

    def __init__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None,
                 target_col: str = 'SalePrice'):
        """
        Initialize feature engineer.

        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
            target_col: Target column name
        """
        self.train_data = train_data.copy()
        self.test_data = test_data.copy() if test_data is not None else None
        self.target_col = target_col
        self.feature_transformers = {}
        self.created_features = []

    def create_area_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create area-related features.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        # Total area features
        area_features = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']
        existing_area_features = [f for f in area_features if f in train_enhanced.columns]

        if len(existing_area_features) >= 2:
            # Total area above ground
            if '1stFlrSF' in train_enhanced.columns and '2ndFlrSF' in train_enhanced.columns:
                train_enhanced['TotalAboveGrSF'] = train_enhanced['1stFlrSF'] + train_enhanced['2ndFlrSF']
                self.created_features.append('TotalAboveGrSF')

                if test_enhanced is not None:
                    test_enhanced['TotalAboveGrSF'] = test_enhanced['1stFlrSF'] + test_enhanced['2ndFlrSF']

            # Total square footage
            if 'TotalBsmtSF' in train_enhanced.columns and 'GrLivArea' in train_enhanced.columns:
                train_enhanced['TotalSF'] = train_enhanced['TotalBsmtSF'] + train_enhanced['GrLivArea']
                self.created_features.append('TotalSF')

                if test_enhanced is not None:
                    test_enhanced['TotalSF'] = test_enhanced['TotalBsmtSF'] + test_enhanced['GrLivArea']

        # Porch area features
        porch_features = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        existing_porch_features = [f for f in porch_features if f in train_enhanced.columns]

        if len(existing_porch_features) >= 2:
            train_enhanced['TotalPorchSF'] = train_enhanced[existing_porch_features].sum(axis=1)
            self.created_features.append('TotalPorchSF')

            if test_enhanced is not None:
                test_enhanced['TotalPorchSF'] = test_enhanced[existing_porch_features].sum(axis=1)

        return train_enhanced, test_enhanced

    def create_age_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create age-related features.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        # House age at time of sale
        if 'YearBuilt' in train_enhanced.columns and 'YrSold' in train_enhanced.columns:
            train_enhanced['HouseAge'] = train_enhanced['YrSold'] - train_enhanced['YearBuilt']
            self.created_features.append('HouseAge')

            if test_enhanced is not None:
                test_enhanced['HouseAge'] = test_enhanced['YrSold'] - test_enhanced['YearBuilt']

        # Years since remodel
        if 'YearRemodAdd' in train_enhanced.columns and 'YrSold' in train_enhanced.columns:
            train_enhanced['YearsSinceRemodel'] = train_enhanced['YrSold'] - train_enhanced['YearRemodAdd']
            self.created_features.append('YearsSinceRemodel')

            if test_enhanced is not None:
                test_enhanced['YearsSinceRemodel'] = test_enhanced['YrSold'] - test_enhanced['YearRemodAdd']

        # Garage age
        if 'GarageYrBlt' in train_enhanced.columns and 'YrSold' in train_enhanced.columns:
            # Handle missing garage years
            train_enhanced['GarageAge'] = train_enhanced['YrSold'] - train_enhanced['GarageYrBlt'].fillna(train_enhanced['YearBuilt'])
            self.created_features.append('GarageAge')

            if test_enhanced is not None:
                test_enhanced['GarageAge'] = test_enhanced['YrSold'] - test_enhanced['GarageYrBlt'].fillna(test_enhanced['YearBuilt'])

        return train_enhanced, test_enhanced

    def create_bathroom_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create bathroom-related features.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        # Total bathrooms
        bathroom_features = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        existing_bathroom_features = [f for f in bathroom_features if f in train_enhanced.columns]

        if len(existing_bathroom_features) >= 2:
            # Total full bathrooms
            full_bath_features = [f for f in ['FullBath', 'BsmtFullBath'] if f in train_enhanced.columns]
            if len(full_bath_features) >= 1:
                train_enhanced['TotalFullBath'] = train_enhanced[full_bath_features].sum(axis=1)
                self.created_features.append('TotalFullBath')

                if test_enhanced is not None:
                    test_enhanced['TotalFullBath'] = test_enhanced[full_bath_features].sum(axis=1)

            # Total half bathrooms
            half_bath_features = [f for f in ['HalfBath', 'BsmtHalfBath'] if f in train_enhanced.columns]
            if len(half_bath_features) >= 1:
                train_enhanced['TotalHalfBath'] = train_enhanced[half_bath_features].sum(axis=1)
                self.created_features.append('TotalHalfBath')

                if test_enhanced is not None:
                    test_enhanced['TotalHalfBath'] = test_enhanced[half_bath_features].sum(axis=1)

            # Total bathrooms (full + 0.5 * half)
            if 'TotalFullBath' in train_enhanced.columns and 'TotalHalfBath' in train_enhanced.columns:
                train_enhanced['TotalBathrooms'] = train_enhanced['TotalFullBath'] + 0.5 * train_enhanced['TotalHalfBath']
                self.created_features.append('TotalBathrooms')

                if test_enhanced is not None:
                    test_enhanced['TotalBathrooms'] = test_enhanced['TotalFullBath'] + 0.5 * test_enhanced['TotalHalfBath']

        return train_enhanced, test_enhanced

    def create_quality_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create quality-related features.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        # Overall quality × condition interaction
        if 'OverallQual' in train_enhanced.columns and 'OverallCond' in train_enhanced.columns:
            train_enhanced['QualCondProduct'] = train_enhanced['OverallQual'] * train_enhanced['OverallCond']
            self.created_features.append('QualCondProduct')

            if test_enhanced is not None:
                test_enhanced['QualCondProduct'] = test_enhanced['OverallQual'] * test_enhanced['OverallCond']

        # Quality score (average of quality ratings)
        quality_features = ['OverallQual', 'ExterQual', 'KitchenQual', 'BsmtQual', 'HeatingQC']

        # Convert quality ratings to numeric
        quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

        for feature in quality_features:
            if feature in train_enhanced.columns and train_enhanced[feature].dtype == 'object':
                # Map quality ratings to numbers
                train_enhanced[f'{feature}_Numeric'] = train_enhanced[feature].map(quality_mapping)
                self.created_features.append(f'{feature}_Numeric')

                if test_enhanced is not None:
                    test_enhanced[f'{feature}_Numeric'] = test_enhanced[feature].map(quality_mapping)

        return train_enhanced, test_enhanced

    def create_interaction_features(self, important_features: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create interaction features between important variables.

        Args:
            important_features: List of important features to create interactions for

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        if important_features is None:
            # Default important features for interactions
            important_features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageCars']

        # Filter to existing features
        existing_features = [f for f in important_features if f in train_enhanced.columns]
        numerical_features = [f for f in existing_features if train_enhanced[f].dtype in ['int64', 'float64']]

        # Create pairwise interactions for top numerical features (limit to avoid explosion)
        for i, feature1 in enumerate(numerical_features[:3]):
            for feature2 in numerical_features[i+1:4]:
                interaction_name = f'{feature1}_x_{feature2}'
                train_enhanced[interaction_name] = train_enhanced[feature1] * train_enhanced[feature2]
                self.created_features.append(interaction_name)

                if test_enhanced is not None:
                    test_enhanced[interaction_name] = test_enhanced[feature1] * test_enhanced[feature2]

        return train_enhanced, test_enhanced

    def encode_categorical_features(self, encoding_method: str = 'onehot',
                                  high_cardinality_threshold: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features.

        Args:
            encoding_method: 'onehot', 'label', or 'target'
            high_cardinality_threshold: Threshold for high cardinality features

        Returns:
            Tuple of (encoded_train, encoded_test)
        """
        train_encoded = self.train_data.copy()
        test_encoded = self.test_data.copy() if self.test_data is not None else None

        # Get categorical columns
        categorical_columns = train_encoded.select_dtypes(include=['object']).columns.tolist()

        for column in categorical_columns:
            unique_count = train_encoded[column].nunique()

            if unique_count <= high_cardinality_threshold:
                if encoding_method == 'onehot':
                    # One-hot encoding
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_features = encoder.fit_transform(train_encoded[[column]])

                    # Create feature names
                    feature_names = [f'{column}_{cat}' for cat in encoder.categories_[0]]

                    # Add encoded features to dataframe
                    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=train_encoded.index)
                    train_encoded = pd.concat([train_encoded, encoded_df], axis=1)

                    # Apply to test data
                    if test_encoded is not None:
                        test_features = encoder.transform(test_encoded[[column]])
                        test_encoded_df = pd.DataFrame(test_features, columns=feature_names, index=test_encoded.index)
                        test_encoded = pd.concat([test_encoded, test_encoded_df], axis=1)

                    # Store encoder
                    self.feature_transformers[f'{column}_onehot'] = encoder
                    self.created_features.extend(feature_names)

                elif encoding_method == 'label':
                    # Label encoding
                    encoder = LabelEncoder()
                    train_encoded[f'{column}_encoded'] = encoder.fit_transform(train_encoded[column].astype(str))

                    if test_encoded is not None:
                        # Handle unknown categories in test set
                        test_encoded[f'{column}_encoded'] = test_encoded[column].map(
                            dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                        ).fillna(-1)

                    self.feature_transformers[f'{column}_label'] = encoder
                    self.created_features.append(f'{column}_encoded')

                # Remove original categorical column
                train_encoded = train_encoded.drop(column, axis=1)
                if test_encoded is not None:
                    test_encoded = test_encoded.drop(column, axis=1)

            else:
                # High cardinality - use target encoding or drop
                print(f"Warning: High cardinality feature '{column}' ({unique_count} unique values) - consider target encoding")
                # For now, convert to string and keep
                train_encoded[column] = train_encoded[column].astype(str)
                if test_encoded is not None:
                    test_encoded[column] = test_encoded[column].astype(str)

        return train_encoded, test_encoded

    def scale_features(self, numerical_features: List[str] = None,
                      scaling_method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features.

        Args:
            numerical_features: List of features to scale (default: all numerical)
            scaling_method: 'standard', 'robust', or 'minmax'

        Returns:
            Tuple of (scaled_train, scaled_test)
        """
        train_scaled = self.train_data.copy()
        test_scaled = self.test_data.copy() if self.test_data is not None else None

        if numerical_features is None:
            numerical_features = train_scaled.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target and ID columns
            numerical_features = [col for col in numerical_features if col not in [self.target_col, 'Id']]

        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()

        # Fit scaler on training data and transform both train and test
        train_scaled[numerical_features] = scaler.fit_transform(train_scaled[numerical_features])
        self.feature_transformers['scaler'] = scaler

        if test_scaled is not None:
            test_scaled[numerical_features] = scaler.transform(test_scaled[numerical_features])

        return train_scaled, test_scaled

    def apply_feature_engineering(self, include_interactions: bool = True,
                                encoding_method: str = 'onehot',
                                scaling_method: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply complete feature engineering pipeline.

        Args:
            include_interactions: Whether to create interaction features
            encoding_method: Categorical encoding method
            scaling_method: Numerical scaling method

        Returns:
            Tuple of (engineered_train, engineered_test)
        """
        # Start with copies of original data
        train_engineered = self.train_data.copy()
        test_engineered = self.test_data.copy() if self.test_data is not None else None

        # Create new features
        train_engineered, test_engineered = self.create_area_features()
        self.train_data, self.test_data = train_engineered, test_engineered

        train_engineered, test_engineered = self.create_age_features()
        self.train_data, self.test_data = train_engineered, test_engineered

        train_engineered, test_engineered = self.create_bathroom_features()
        self.train_data, self.test_data = train_engineered, test_engineered

        train_engineered, test_engineered = self.create_quality_features()
        self.train_data, self.test_data = train_engineered, test_engineered

        if include_interactions:
            train_engineered, test_engineered = self.create_interaction_features()
            self.train_data, self.test_data = train_engineered, test_engineered

        # Encode categorical features
        train_engineered, test_engineered = self.encode_categorical_features(encoding_method)
        self.train_data, self.test_data = train_engineered, test_engineered

        # Scale numerical features
        train_engineered, test_engineered = self.scale_features(scaling_method=scaling_method)

        return train_engineered, test_engineered

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature engineering operations.

        Returns:
            Feature engineering summary
        """
        return {
            'original_features': self.train_data.shape[1] - len(self.created_features),
            'created_features': self.created_features,
            'total_created': len(self.created_features),
            'final_feature_count': self.train_data.shape[1],
            'transformers_fitted': list(self.feature_transformers.keys())
        }
```

**Test**: Feature engineering
```python
python -c "
from src.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np

# Create sample data with house-like features
sample_data = pd.DataFrame({
    'Id': range(1, 101),
    'SalePrice': np.random.normal(200000, 50000, 100),
    'GrLivArea': np.random.normal(1500, 300, 100),
    'TotalBsmtSF': np.random.normal(1000, 200, 100),
    'YearBuilt': np.random.randint(1950, 2020, 100),
    'YrSold': np.random.randint(2006, 2011, 100),
    'OverallQual': np.random.randint(1, 11, 100),
    'OverallCond': np.random.randint(1, 11, 100),
    'Neighborhood': np.random.choice(['A', 'B', 'C'], 100)
})

engineer = FeatureEngineer(sample_data)
train_engineered, _ = engineer.apply_feature_engineering()
summary = engineer.get_feature_summary()

print('✓ Feature engineering module working correctly')
print(f'Original features: {summary[\"original_features\"]}')
print(f'Created features: {summary[\"total_created\"]}')
print(f'Final feature count: {summary[\"final_feature_count\"]}')
"
```

### 3.4 Data Splitting and Pipeline Creation

#### 3.4.1 Create Data Preparation Pipeline
Create `src/data_pipeline.py`:
```python
"""
Complete data preprocessing pipeline for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

from config.settings import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE
from src.data_cleaning import DataCleaner
from src.missing_value_handler import MissingValueHandler
from src.feature_engineering import FeatureEngineer
from src.utils import save_data, ensure_dir_exists


class DataPreprocessingPipeline:
    """Complete data preprocessing pipeline."""

    def __init__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None,
                 target_col: str = 'SalePrice'):
        """
        Initialize preprocessing pipeline.

        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
            target_col: Target column name
        """
        self.original_train = train_data.copy()
        self.original_test = test_data.copy() if test_data is not None else None
        self.target_col = target_col

        # Pipeline components
        self.cleaner = None
        self.missing_handler = None
        self.feature_engineer = None

        # Processing results
        self.processed_train = None
        self.processed_test = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        # Pipeline configuration
        self.pipeline_config = {
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'outlier_factor': 1.5,
            'encoding_method': 'onehot',
            'scaling_method': 'robust',
            'include_interactions': True,
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE
        }

        # Processing log
        self.processing_log = []

    def log_step(self, step: str, details: str, data_shape: Tuple[int, int]) -> None:
        """
        Log processing step.

        Args:
            step: Processing step name
            details: Step details
            data_shape: Data shape after step
        """
        self.processing_log.append({
            'step': step,
            'details': details,
            'data_shape': data_shape,
            'timestamp': pd.Timestamp.now()
        })

    def clean_data(self, remove_outliers: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean the data.

        Args:
            remove_outliers: Whether to remove outliers

        Returns:
            Tuple of (cleaned_train, cleaned_test)
        """
        self.cleaner = DataCleaner(self.original_train, self.original_test)

        # Remove duplicates
        train_clean, test_clean = self.cleaner.remove_duplicates()
        self.log_step('remove_duplicates', 'Removed duplicate records', train_clean.shape)

        # Standardize column names
        self.cleaner.standardize_column_names()
        self.log_step('standardize_columns', 'Standardized column names', train_clean.shape)

        # Convert data types
        self.cleaner.convert_data_types()
        self.log_step('convert_types', 'Converted data types', train_clean.shape)

        # Handle outliers
        if remove_outliers:
            train_clean, outlier_indices = self.cleaner.handle_outliers(
                method=self.pipeline_config['outlier_method'],
                factor=self.pipeline_config['outlier_factor']
            )
            self.log_step('remove_outliers', f'Removed {len(outlier_indices)} outliers', train_clean.shape)

        return train_clean, test_clean

    def handle_missing_values(self, train_data: pd.DataFrame,
                            test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values.

        Args:
            train_data: Training data
            test_data: Test data

        Returns:
            Tuple of (imputed_train, imputed_test)
        """
        self.missing_handler = MissingValueHandler(train_data, test_data)

        # Analyze missing patterns
        missing_analysis = self.missing_handler.analyze_missing_patterns()
        self.log_step('analyze_missing', f'Analyzed missing patterns', train_data.shape)

        # Create and apply imputation strategy
        strategies = self.missing_handler.create_imputation_strategy(missing_analysis)
        train_imputed, test_imputed = self.missing_handler.apply_imputation(strategies)

        # Validate imputation
        validation_results = self.missing_handler.validate_imputation(train_imputed, test_imputed)
        self.log_step('handle_missing', f'Applied {len(strategies)} imputation strategies', train_imputed.shape)

        return train_imputed, test_imputed

    def engineer_features(self, train_data: pd.DataFrame,
                         test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Engineer features.

        Args:
            train_data: Training data
            test_data: Test data

        Returns:
            Tuple of (engineered_train, engineered_test)
        """
        self.feature_engineer = FeatureEngineer(train_data, test_data, self.target_col)

        train_engineered, test_engineered = self.feature_engineer.apply_feature_engineering(
            include_interactions=self.pipeline_config['include_interactions'],
            encoding_method=self.pipeline_config['encoding_method'],
            scaling_method=self.pipeline_config['scaling_method']
        )

        feature_summary = self.feature_engineer.get_feature_summary()
        self.log_step('engineer_features',
                     f'Created {feature_summary["total_created"]} new features',
                     train_engineered.shape)

        return train_engineered, test_engineered

    def split_data(self, train_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split training data into train/validation sets.

        Args:
            train_data: Processed training data

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Separate features and target
        if self.target_col in train_data.columns:
            X = train_data.drop(self.target_col, axis=1)
            y = train_data[self.target_col]
        else:
            raise ValueError(f"Target column '{self.target_col}' not found in training data")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.pipeline_config['test_size'],
            random_state=self.pipeline_config['random_state'],
            stratify=None  # For regression
        )

        self.log_step('split_data',
                     f'Split into train ({len(X_train)}) and validation ({len(X_val)})',
                     X_train.shape)

        return X_train, X_val, y_train, y_val

    def run_pipeline(self, save_processed: bool = True) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.

        Args:
            save_processed: Whether to save processed data

        Returns:
            Dictionary with pipeline results
        """
        print("Starting data preprocessing pipeline...")

        # Step 1: Clean data
        print("Step 1: Cleaning data...")
        train_clean, test_clean = self.clean_data(self.pipeline_config['remove_outliers'])

        # Step 2: Handle missing values
        print("Step 2: Handling missing values...")
        train_imputed, test_imputed = self.handle_missing_values(train_clean, test_clean)

        # Step 3: Engineer features
        print("Step 3: Engineering features...")
        train_engineered, test_engineered = self.engineer_features(train_imputed, test_imputed)

        # Step 4: Split data
        print("Step 4: Splitting data...")
        X_train, X_val, y_train, y_val = self.split_data(train_engineered)

        # Store results
        self.processed_train = train_engineered
        self.processed_test = test_engineered
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        # Save processed data
        if save_processed:
            self._save_processed_data()

        # Create results summary
        results = {
            'original_train_shape': self.original_train.shape,
            'original_test_shape': self.original_test.shape if self.original_test is not None else None,
            'processed_train_shape': self.processed_train.shape,
            'processed_test_shape': self.processed_test.shape if self.processed_test is not None else None,
            'X_train_shape': self.X_train.shape,
            'X_val_shape': self.X_val.shape,
            'feature_count': self.X_train.shape[1],
            'processing_steps': len(self.processing_log),
            'pipeline_config': self.pipeline_config
        }

        print("Pipeline completed successfully!")
        print(f"Final feature count: {results['feature_count']}")
        print(f"Training samples: {results['X_train_shape'][0]}")
        print(f"Validation samples: {results['X_val_shape'][0]}")

        return results

    def _save_processed_data(self) -> None:
        """Save processed data to files."""
        ensure_dir_exists(PROCESSED_DATA_DIR)

        # Save processed datasets
        if self.processed_train is not None:
            save_data(self.processed_train, PROCESSED_DATA_DIR / 'train_processed.csv')

        if self.processed_test is not None:
            save_data(self.processed_test, PROCESSED_DATA_DIR / 'test_processed.csv')

        # Save train/validation splits
        if all(x is not None for x in [self.X_train, self.X_val, self.y_train, self.y_val]):
            save_data(self.X_train, PROCESSED_DATA_DIR / 'X_train.csv')
            save_data(self.X_val, PROCESSED_DATA_DIR / 'X_val.csv')
            save_data(self.y_train.to_frame(), PROCESSED_DATA_DIR / 'y_train.csv')
            save_data(self.y_val.to_frame(), PROCESSED_DATA_DIR / 'y_val.csv')

        # Save pipeline components
        pipeline_artifacts = {
            'cleaner': self.cleaner,
            'missing_handler': self.missing_handler,
            'feature_engineer': self.feature_engineer,
            'pipeline_config': self.pipeline_config,
            'processing_log': self.processing_log
        }

        joblib.dump(pipeline_artifacts, PROCESSED_DATA_DIR / 'pipeline_artifacts.pkl')

        print(f"Processed data saved to: {PROCESSED_DATA_DIR}")

    def get_processing_summary(self) -> str:
        """
        Get a text summary of the processing pipeline.

        Returns:
            Processing summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("DATA PREPROCESSING PIPELINE SUMMARY")
        summary.append("=" * 60)

        # Original data info
        summary.append(f"\nORIGINAL DATA:")
        summary.append(f"Training data shape: {self.original_train.shape}")
        if self.original_test is not None:
            summary.append(f"Test data shape: {self.original_test.shape}")

        # Processing steps
        summary.append(f"\nPROCESSING STEPS PERFORMED:")
        for i, log_entry in enumerate(self.processing_log, 1):
            summary.append(f"{i}. {log_entry['step']}: {log_entry['details']}")
            summary.append(f"   Result shape: {log_entry['data_shape']}")

        # Final results
        if self.X_train is not None:
            summary.append(f"\nFINAL RESULTS:")
            summary.append(f"Feature count: {self.X_train.shape[1]}")
            summary.append(f"Training samples: {self.X_train.shape[0]}")
            summary.append(f"Validation samples: {self.X_val.shape[0]}")

        # Configuration used
        summary.append(f"\nPIPELINE CONFIGURATION:")
        for key, value in self.pipeline_config.items():
            summary.append(f"  {key}: {value}")

        return "\n".join(summary)

    def load_processed_data(self, data_dir: Path = None) -> Dict[str, pd.DataFrame]:
        """
        Load previously processed data.

        Args:
            data_dir: Directory containing processed data

        Returns:
            Dictionary with loaded datasets
        """
        if data_dir is None:
            data_dir = PROCESSED_DATA_DIR

        datasets = {}

        # Load datasets if they exist
        file_mappings = {
            'X_train': 'X_train.csv',
            'X_val': 'X_val.csv',
            'y_train': 'y_train.csv',
            'y_val': 'y_val.csv',
            'train_processed': 'train_processed.csv',
            'test_processed': 'test_processed.csv'
        }

        for dataset_name, filename in file_mappings.items():
            filepath = data_dir / filename
            if filepath.exists():
                datasets[dataset_name] = pd.read_csv(filepath)
                if dataset_name in ['y_train', 'y_val']:
                    # Convert back to series
                    datasets[dataset_name] = datasets[dataset_name].iloc[:, 0]

        return datasets
```

**Test**: Complete pipeline
```python
python -c "
from src.data_pipeline import DataPreprocessingPipeline
import pandas as pd
import numpy as np

# Create sample house data
np.random.seed(42)
sample_data = pd.DataFrame({
    'Id': range(1, 501),
    'SalePrice': np.random.normal(200000, 50000, 500),
    'GrLivArea': np.random.normal(1500, 300, 500),
    'TotalBsmtSF': np.random.normal(1000, 200, 500),
    'YearBuilt': np.random.randint(1950, 2020, 500),
    'YrSold': np.random.randint(2006, 2011, 500),
    'OverallQual': np.random.randint(1, 11, 500),
    'Neighborhood': np.random.choice(['A', 'B', 'C', 'D'], 500),
    'MissingFeature': [None if i % 4 == 0 else np.random.normal(100, 20) for i in range(500)]
})

# Create test data (without target)
test_data = sample_data.drop('SalePrice', axis=1).copy()

# Run pipeline
pipeline = DataPreprocessingPipeline(sample_data, test_data)
results = pipeline.run_pipeline(save_processed=False)

print('✓ Complete preprocessing pipeline working correctly')
print(f'Original features: {sample_data.shape[1]}')
print(f'Final features: {results[\"feature_count\"]}')
print(f'Training samples: {results[\"X_train_shape\"][0]}')
print(f'Validation samples: {results[\"X_val_shape\"][0]}')
"
```

### 3.5 Data Processing Notebook

#### 3.5.1 Create Processing Notebook
Create `notebooks/02_data_processing_feature_engineering.ipynb`:

```python
# Cell 1: Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from src.data_loader import load_kaggle_data, validate_dataset
from src.data_pipeline import DataPreprocessingPipeline
from src.data_cleaning import DataCleaner
from src.missing_value_handler import MissingValueHandler
from src.feature_engineering import FeatureEngineer

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Data Processing Environment Setup Complete")
```

```python
# Cell 2: Load Data
try:
    train_data, test_data = load_kaggle_data()
    print("✓ Kaggle dataset loaded successfully")
except FileNotFoundError:
    print("Creating sample data for demonstration...")
    # Create comprehensive sample data
    np.random.seed(42)
    train_data = pd.DataFrame({
        'Id': range(1, 1001),
        'SalePrice': np.random.lognormal(12, 0.4, 1000),
        'GrLivArea': np.random.normal(1500, 400, 1000),
        'TotalBsmtSF': np.random.normal(1000, 300, 1000),
        '1stFlrSF': np.random.normal(800, 200, 1000),
        '2ndFlrSF': np.random.normal(700, 250, 1000),
        'YearBuilt': np.random.randint(1950, 2020, 1000),
        'YrSold': np.random.randint(2006, 2011, 1000),
        'YearRemodAdd': np.random.randint(1950, 2020, 1000),
        'OverallQual': np.random.randint(1, 11, 1000),
        'OverallCond': np.random.randint(1, 11, 1000),
        'Neighborhood': np.random.choice(['CollgCr', 'OldTown', 'Edwards', 'Somerst'], 1000),
        'ExterQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], 1000),
        'FullBath': np.random.randint(1, 4, 1000),
        'HalfBath': np.random.randint(0, 3, 1000),
        'BsmtFullBath': np.random.randint(0, 3, 1000),
        'MissingFeature': [None if i % 5 == 0 else np.random.normal(50, 10) for i in range(1000)]
    })
    test_data = train_data.drop('SalePrice', axis=1).sample(500).reset_index(drop=True)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
```

```python
# Cell 3: Initialize and Run Complete Pipeline
print("Initializing preprocessing pipeline...")
pipeline = DataPreprocessingPipeline(train_data, test_data)

# Run the complete pipeline
print("\nRunning complete preprocessing pipeline...")
results = pipeline.run_pipeline(save_processed=True)

# Display results
print("\n" + "="*60)
print("PIPELINE RESULTS SUMMARY")
print("="*60)
print(f"Original training shape: {results['original_train_shape']}")
print(f"Processed training shape: {results['processed_train_shape']}")
print(f"Final feature count: {results['feature_count']}")
print(f"Training samples: {results['X_train_shape'][0]}")
print(f"Validation samples: {results['X_val_shape'][0]}")
```

```python
# Cell 4: Analyze Feature Engineering Results
feature_summary = pipeline.feature_engineer.get_feature_summary()

print("FEATURE ENGINEERING SUMMARY:")
print(f"Original features: {feature_summary['original_features']}")
print(f"Created features: {feature_summary['total_created']}")
print(f"Final feature count: {feature_summary['final_feature_count']}")

print(f"\nCreated features:")
for feature in feature_summary['created_features'][:10]:  # Show first 10
    print(f"  - {feature}")
if len(feature_summary['created_features']) > 10:
    print(f"  ... and {len(feature_summary['created_features']) - 10} more")
```

```python
# Cell 5: Visualize Processing Results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Target distribution before and after processing
if 'SalePrice' in train_data.columns and pipeline.y_train is not None:
    # Original target distribution
    axes[0,0].hist(train_data['SalePrice'], bins=50, alpha=0.7, color='skyblue', label='Original')
    axes[0,0].set_title('Original SalePrice Distribution')
    axes[0,0].set_xlabel('Sale Price')
    axes[0,0].set_ylabel('Frequency')

    # Processed target distribution
    axes[0,1].hist(pipeline.y_train, bins=50, alpha=0.7, color='lightgreen', label='Processed')
    axes[0,1].set_title('Processed SalePrice Distribution')
    axes[0,1].set_xlabel('Sale Price')
    axes[0,1].set_ylabel('Frequency')

# Feature count comparison
categories = ['Original', 'After Engineering']
feature_counts = [feature_summary['original_features'], feature_summary['final_feature_count']]
axes[1,0].bar(categories, feature_counts, color=['coral', 'lightblue'])
axes[1,0].set_title('Feature Count Comparison')
axes[1,0].set_ylabel('Number of Features')

# Missing values before and after
missing_before = train_data.isnull().sum().sum()
missing_after = pipeline.processed_train.isnull().sum().sum()
categories = ['Before Processing', 'After Processing']
missing_counts = [missing_before, missing_after]
axes[1,1].bar(categories, missing_counts, color=['red', 'green'])
axes[1,1].set_title('Missing Values Comparison')
axes[1,1].set_ylabel('Total Missing Values')

plt.tight_layout()
plt.show()
```

```python
# Cell 6: Display Processing Log
print("DETAILED PROCESSING LOG:")
print("="*60)
processing_summary = pipeline.get_processing_summary()
print(processing_summary)
```

### 3.6 Testing and Validation

Create `tests/test_phase3.py`:
```python
"""
Tests for Phase 3: Data Processing & Feature Engineering
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('..')

from src.data_cleaning import DataCleaner
from src.missing_value_handler import MissingValueHandler
from src.feature_engineering import FeatureEngineer
from src.data_pipeline import DataPreprocessingPipeline


class TestPhase3(unittest.TestCase):
    """Test Phase 3 functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Id': range(1, 101),
            'SalePrice': np.random.normal(200000, 50000, 100),
            'GrLivArea': np.random.normal(1500, 300, 100),
            'TotalBsmtSF': np.random.normal(1000, 200, 100),
            '1stFlrSF': np.random.normal(800, 150, 100),
            '2ndFlrSF': np.random.normal(700, 200, 100),
            'YearBuilt': np.random.randint(1950, 2020, 100),
            'YrSold': np.random.randint(2006, 2011, 100),
            'OverallQual': np.random.randint(1, 11, 100),
            'OverallCond': np.random.randint(1, 11, 100),
            'Neighborhood': np.random.choice(['A', 'B', 'C'], 100),
            'MissingFeature': [None if i % 3 == 0 else i for i in range(100)]
        })

        # Add some duplicates and outliers
        self.sample_data.loc[99] = self.sample_data.loc[98]  # Duplicate
        self.sample_data.loc[97, 'GrLivArea'] = 10000  # Outlier

        self.test_data = self.sample_data.drop('SalePrice', axis=1)

    def test_data_cleaner(self):
        """Test data cleaning functionality."""
        cleaner = DataCleaner(self.sample_data, self.test_data)

        # Test duplicate removal
        train_clean, test_clean = cleaner.remove_duplicates()
        self.assertLess(len(train_clean), len(self.sample_data))

        # Test outlier handling
        train_no_outliers, outlier_indices = cleaner.handle_outliers()
        self.assertIsInstance(outlier_indices, list)

        # Test column standardization
        cleaner.standardize_column_names()
        # Check that column names are lowercase
        for col in cleaner.train_data.columns:
            self.assertEqual(col, col.lower())

    def test_missing_value_handler(self):
        """Test missing value handling."""
        handler = MissingValueHandler(self.sample_data, self.test_data)

        # Test missing pattern analysis
        analysis = handler.analyze_missing_patterns()
        self.assertIsInstance(analysis, dict)
        self.assertIn('train_missing_counts', analysis)

        # Test imputation strategy creation
        strategies = handler.create_imputation_strategy(analysis)
        self.assertIsInstance(strategies, dict)

        # Test imputation application
        train_imputed, test_imputed = handler.apply_imputation()
        initial_missing = self.sample_data.isnull().sum().sum()
        final_missing = train_imputed.isnull().sum().sum()
        self.assertLessEqual(final_missing, initial_missing)

    def test_feature_engineer(self):
        """Test feature engineering."""
        engineer = FeatureEngineer(self.sample_data, self.test_data)

        # Test area feature creation
        train_areas, test_areas = engineer.create_area_features()
        if 'TotalBsmtSF' in train_areas.columns and 'GrLivArea' in train_areas.columns:
            self.assertIn('TotalSF', train_areas.columns)

        # Test age feature creation
        train_ages, test_ages = engineer.create_age_features()
        if 'YearBuilt' in train_ages.columns and 'YrSold' in train_ages.columns:
            self.assertIn('HouseAge', train_ages.columns)

        # Test quality feature creation
        train_quality, test_quality = engineer.create_quality_features()
        if 'OverallQual' in train_quality.columns and 'OverallCond' in train_quality.columns:
            self.assertIn('QualCondProduct', train_quality.columns)

        # Test complete feature engineering
        train_engineered, test_engineered = engineer.apply_feature_engineering()
        self.assertGreater(train_engineered.shape[1], self.sample_data.shape[1])

    def test_complete_pipeline(self):
        """Test complete preprocessing pipeline."""
        pipeline = DataPreprocessingPipeline(self.sample_data, self.test_data)

        # Run pipeline
        results = pipeline.run_pipeline(save_processed=False)

        # Validate results
        self.assertIsInstance(results, dict)
        self.assertIn('X_train_shape', results)
        self.assertIn('X_val_shape', results)
        self.assertIn('feature_count', results)

        # Check that data was split correctly
        self.assertIsNotNone(pipeline.X_train)
        self.assertIsNotNone(pipeline.X_val)
        self.assertIsNotNone(pipeline.y_train)
        self.assertIsNotNone(pipeline.y_val)

        # Check feature count increased
        original_features = self.sample_data.shape[1] - 1  # Exclude target
        final_features = results['feature_count']
        self.assertGreater(final_features, original_features)

    def test_pipeline_configuration(self):
        """Test pipeline configuration options."""
        # Test with different configurations
        pipeline = DataPreprocessingPipeline(self.sample_data, self.test_data)

        # Modify configuration
        pipeline.pipeline_config['remove_outliers'] = False
        pipeline.pipeline_config['include_interactions'] = False

        results = pipeline.run_pipeline(save_processed=False)
        self.assertIsInstance(results, dict)


if __name__ == '__main__':
    unittest.main()
```

**Test**: Run Phase 3 tests
```bash
cd tests
python test_phase3.py
```

## Deliverables
- [ ] Data cleaning module with outlier handling
- [ ] Comprehensive missing value handling system
- [ ] Feature engineering pipeline with new feature creation
- [ ] Complete data preprocessing pipeline
- [ ] Data splitting for train/validation sets
- [ ] Processed datasets saved for modeling
- [ ] Feature scaling and encoding
- [ ] Processing documentation and logs
- [ ] Comprehensive testing suite
- [ ] Data processing notebook with visualizations

## Success Criteria
- All missing values handled appropriately
- New meaningful features created and validated
- Data properly cleaned and preprocessed
- Feature encoding and scaling applied correctly
- Train/validation split performed
- Pipeline runs end-to-end without errors
- All tests pass successfully
- Processed data ready for modeling

## Next Phase
Proceed to **Phase 4: Model Development & Training** with the processed and engineered features.