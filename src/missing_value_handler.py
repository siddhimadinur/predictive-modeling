"""
Missing value handling strategies optimized for housing data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.impute import SimpleImputer, KNNImputer


class MissingValueHandler:
    """Handle missing values with strategies optimized for housing data."""

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
        Analyze missing value patterns specific to housing data.

        Returns:
            Dictionary with missing pattern analysis
        """
        missing_analysis = {}

        # Get missing value counts and percentages
        train_missing = self.train_data.isnull().sum()
        train_missing_pct = (train_missing / len(self.train_data)) * 100

        analysis = {
            'train_missing_counts': train_missing[train_missing > 0].to_dict(),
            'train_missing_percentages': train_missing_pct[train_missing_pct > 0].to_dict(),
            'total_missing': int(train_missing.sum()),
            'missing_percentage_overall': float(train_missing.sum() / (len(self.train_data) * len(self.train_data.columns)) * 100)
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
            'low_missing_columns': low_missing,
            'complete_columns': train_missing_pct[train_missing_pct == 0].index.tolist()
        })

        return analysis

    def create_housing_imputation_strategy(self, missing_analysis: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Create imputation strategy optimized for housing data.

        Args:
            missing_analysis: Missing value analysis (if None, will compute)

        Returns:
            Dictionary mapping columns to imputation strategies
        """
        if missing_analysis is None:
            missing_analysis = self.analyze_missing_patterns()

        strategies = {}

        # Housing-specific imputation strategies
        housing_strategies = {
            # Geographic features - use median
            'longitude': 'median_imputation',
            'latitude': 'median_imputation',

            # Room/space features - use median or KNN
            'total_rooms': 'knn_imputation',
            'total_bedrooms': 'knn_imputation',
            'households': 'knn_imputation',
            'population': 'knn_imputation',

            # Age and condition - use median
            'housing_median_age': 'median_imputation',

            # Economic features - use median
            'median_income': 'median_imputation',
            'median_house_value': 'median_imputation',

            # Categorical features - use mode
            'ocean_proximity': 'mode_imputation'
        }

        # Apply strategies based on missing severity
        for col in missing_analysis['high_missing_columns']:
            if col in housing_strategies:
                strategies[col] = 'drop_column'  # Drop if >50% missing
            else:
                strategies[col] = 'fill_missing_category'

        for col in missing_analysis['medium_missing_columns']:
            strategies[col] = housing_strategies.get(col, 'knn_imputation')

        for col in missing_analysis['low_missing_columns']:
            strategies[col] = housing_strategies.get(col, 'median_imputation')

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
            missing_analysis = self.analyze_missing_patterns()
            strategies = self.create_housing_imputation_strategy(missing_analysis)

        train_imputed = self.train_data.copy()
        test_imputed = self.test_data.copy() if self.test_data is not None else None

        for column, strategy in strategies.items():
            if column not in train_imputed.columns:
                continue

            if strategy == 'drop_column':
                train_imputed = train_imputed.drop(column, axis=1)
                if test_imputed is not None and column in test_imputed.columns:
                    test_imputed = test_imputed.drop(column, axis=1)
                print(f"🗑️ Dropped column: {column}")

            elif strategy == 'median_imputation':
                if train_imputed[column].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy='median')
                    train_imputed[column] = imputer.fit_transform(train_imputed[[column]]).ravel()
                    self.fitted_imputers[column] = imputer

                    if test_imputed is not None and column in test_imputed.columns:
                        test_imputed[column] = imputer.transform(test_imputed[[column]]).ravel()

                    print(f"📊 Median imputation applied to: {column}")

            elif strategy == 'mean_imputation':
                if train_imputed[column].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy='mean')
                    train_imputed[column] = imputer.fit_transform(train_imputed[[column]]).ravel()
                    self.fitted_imputers[column] = imputer

                    if test_imputed is not None and column in test_imputed.columns:
                        test_imputed[column] = imputer.transform(test_imputed[[column]]).ravel()

                    print(f"📊 Mean imputation applied to: {column}")

            elif strategy == 'mode_imputation':
                # Handle categorical features with None values
                if train_imputed[column].dtype == 'object' or train_imputed[column].dtype.name == 'string':
                    # Fill None/NaN with 'Unknown' first, then apply mode imputation
                    train_imputed[column] = train_imputed[column].fillna('Unknown')

                    imputer = SimpleImputer(strategy='most_frequent')
                    train_imputed[column] = imputer.fit_transform(train_imputed[[column]]).ravel()
                    self.fitted_imputers[column] = imputer

                    if test_imputed is not None and column in test_imputed.columns:
                        test_imputed[column] = test_imputed[column].fillna('Unknown')
                        test_imputed[column] = imputer.transform(test_imputed[[column]]).ravel()
                else:
                    # For numerical features, use mode imputation normally
                    imputer = SimpleImputer(strategy='most_frequent')
                    train_imputed[column] = imputer.fit_transform(train_imputed[[column]]).ravel()
                    self.fitted_imputers[column] = imputer

                    if test_imputed is not None and column in test_imputed.columns:
                        test_imputed[column] = imputer.transform(test_imputed[[column]]).ravel()

                print(f"📊 Mode imputation applied to: {column}")

            elif strategy == 'knn_imputation':
                # For KNN imputation, use only numerical columns
                numerical_cols = train_imputed.select_dtypes(include=[np.number]).columns.tolist()

                if column in numerical_cols and len(numerical_cols) > 1:
                    # Use subset of numerical columns to avoid memory issues
                    knn_features = numerical_cols[:min(10, len(numerical_cols))]

                    if column not in knn_features:
                        knn_features = knn_features[:-1] + [column]

                    imputer = KNNImputer(n_neighbors=5)

                    # Fit on selected numerical columns
                    train_subset = train_imputed[knn_features]
                    imputed_values = imputer.fit_transform(train_subset)

                    # Update only the target column
                    col_index = knn_features.index(column)
                    train_imputed[column] = imputed_values[:, col_index]
                    self.fitted_imputers[column] = (imputer, knn_features)

                    if test_imputed is not None and column in test_imputed.columns:
                        test_subset = test_imputed[knn_features]
                        test_imputed_values = imputer.transform(test_subset)
                        test_imputed[column] = test_imputed_values[:, col_index]

                    print(f"🔗 KNN imputation applied to: {column}")

            elif strategy == 'fill_missing_category':
                train_imputed[column] = train_imputed[column].fillna('Missing')
                if test_imputed is not None and column in test_imputed.columns:
                    test_imputed[column] = test_imputed[column].fillna('Missing')

                print(f"📝 Missing category filled for: {column}")

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
            'train_missing_before': int(self.train_data.isnull().sum().sum()),
            'train_missing_after': int(train_imputed.isnull().sum().sum()),
            'train_shape_before': self.train_data.shape,
            'train_shape_after': train_imputed.shape,
            'imputation_successful': train_imputed.isnull().sum().sum() == 0
        }

        if test_imputed is not None and self.test_data is not None:
            validation_results.update({
                'test_missing_before': int(self.test_data.isnull().sum().sum()),
                'test_missing_after': int(test_imputed.isnull().sum().sum()),
                'test_shape_before': self.test_data.shape,
                'test_shape_after': test_imputed.shape
            })

        # Check for any remaining missing values
        remaining_missing_train = train_imputed.isnull().sum()
        remaining_missing_train = remaining_missing_train[remaining_missing_train > 0]

        if len(remaining_missing_train) > 0:
            validation_results['remaining_missing_columns'] = remaining_missing_train.to_dict()
            validation_results['imputation_complete'] = False
        else:
            validation_results['imputation_complete'] = True

        return validation_results


# Test the module
if __name__ == "__main__":
    print("Testing Missing Value Handler...")

    # Create sample data with missing values
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'longitude': [1.0, 2.0, None, 4.0, 5.0, None],
        'total_rooms': [100.0, 200.0, 300.0, None, 500.0, 600.0],
        'ocean_proximity': ['NEAR BAY', 'INLAND', None, 'NEAR BAY', 'INLAND', None],
        'median_house_value': [100000.0, 200000.0, 300000.0, 400000.0, None, 600000.0]
    })

    try:
        handler = MissingValueHandler(sample_data)
        analysis = handler.analyze_missing_patterns()
        strategies = handler.create_housing_imputation_strategy(analysis)
        train_imputed, _ = handler.apply_imputation()

        validation = handler.validate_imputation(train_imputed)

        print("✅ Missing Value Handler working correctly")
        print(f"Strategies created: {len(strategies)}")
        print(f"Missing before: {analysis['total_missing']}")
        print(f"Missing after: {validation['train_missing_after']}")
        print(f"Imputation complete: {validation['imputation_complete']}")

    except Exception as e:
        print(f"❌ Error in Missing Value Handler: {e}")
        import traceback
        traceback.print_exc()