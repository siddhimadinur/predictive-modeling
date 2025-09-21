"""
Data cleaning utilities for California housing price prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from src.utils import save_data, print_data_info


class DataCleaner:
    """Comprehensive data cleaning class optimized for California housing data."""

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
        self.original_train_shape = train_data.shape
        self.original_test_shape = test_data.shape if test_data is not None else None

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
            'test_shape_after': self.test_data.shape if self.test_data is not None else None,
            'timestamp': pd.Timestamp.now()
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

        if train_duplicates_removed > 0 or test_duplicates_removed > 0:
            print(f"🧹 Removed {train_duplicates_removed} train and {test_duplicates_removed} test duplicates")

        return self.train_data, self.test_data

    def handle_outliers(self, columns: List[str] = None, method: str = 'iqr',
                       factor: float = 1.5, max_outliers_pct: float = 5.0) -> Tuple[pd.DataFrame, List[int]]:
        """
        Handle outliers in numerical columns for California housing data.

        Args:
            columns: Columns to process (default: all numerical except target)
            method: Outlier detection method ('iqr', 'zscore')
            factor: Outlier threshold factor
            max_outliers_pct: Maximum percentage of outliers to remove per feature

        Returns:
            Tuple of (cleaned_data, outlier_indices)
        """
        if columns is None:
            # Get numerical columns, exclude Id and target
            numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numerical_cols if col not in ['Id', 'median_house_value', 'SalePrice']]

        outlier_indices = []
        outlier_summary = {}

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

                # Only remove outliers if they're not too many
                outlier_pct = len(outliers) / len(self.train_data) * 100
                if outlier_pct <= max_outliers_pct:
                    outlier_indices.extend(outliers)
                    outlier_summary[col] = {
                        'count': len(outliers),
                        'percentage': outlier_pct,
                        'removed': True
                    }
                else:
                    outlier_summary[col] = {
                        'count': len(outliers),
                        'percentage': outlier_pct,
                        'removed': False,
                        'reason': f'Too many outliers ({outlier_pct:.1f}% > {max_outliers_pct}%)'
                    }

        # Remove outliers from training data only
        outlier_indices = list(set(outlier_indices))
        initial_count = len(self.train_data)

        if outlier_indices:
            self.train_data = self.train_data.drop(outlier_indices)
            removed_count = initial_count - len(self.train_data)

            self.log_operation(
                'handle_outliers',
                f'Removed {removed_count} outlier records using {method} method (factor={factor})'
            )

            print(f"🎯 Outlier removal summary:")
            for col, summary in outlier_summary.items():
                status = "✅ Removed" if summary['removed'] else "⚠️ Kept"
                print(f"  {col}: {summary['count']} outliers ({summary['percentage']:.1f}%) - {status}")

        return self.train_data, outlier_indices

    def standardize_column_names(self) -> None:
        """Standardize column names for consistency."""
        def clean_column_name(name):
            # Keep California housing column names mostly as-is since they're already clean
            name = str(name).strip()
            # Replace spaces with underscores
            name = name.replace(' ', '_').replace('-', '_')
            # Remove special characters except underscores
            name = ''.join(char if char.isalnum() or char == '_' else '' for char in name)
            # Handle consecutive underscores
            while '__' in name:
                name = name.replace('__', '_')
            return name.strip('_')

        # Create mapping of old to new names
        column_mapping = {col: clean_column_name(col) for col in self.train_data.columns}

        # Only rename if there are actual changes
        changes_needed = {old: new for old, new in column_mapping.items() if old != new}

        if changes_needed:
            self.train_data = self.train_data.rename(columns=column_mapping)

            if self.test_data is not None:
                test_mapping = {col: clean_column_name(col) for col in self.test_data.columns}
                self.test_data = self.test_data.rename(columns=test_mapping)

            self.log_operation('standardize_column_names', f'Renamed {len(changes_needed)} columns')
            print(f"📝 Standardized {len(changes_needed)} column names")
        else:
            print("✅ Column names already clean")

    def validate_data_ranges(self) -> Dict[str, Any]:
        """
        Validate data ranges for California housing features.

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        # Define expected ranges for California housing features
        expected_ranges = {
            'longitude': {'min': -125.0, 'max': -114.0, 'description': 'California longitude'},
            'latitude': {'min': 32.0, 'max': 42.5, 'description': 'California latitude'},
            'housing_median_age': {'min': 1.0, 'max': 52.0, 'description': 'Housing age in years'},
            'total_rooms': {'min': 1.0, 'max': 50000.0, 'description': 'Total rooms count'},
            'total_bedrooms': {'min': 1.0, 'max': 7000.0, 'description': 'Total bedrooms count'},
            'population': {'min': 1.0, 'max': 50000.0, 'description': 'Block group population'},
            'households': {'min': 1.0, 'max': 7000.0, 'description': 'Number of households'},
            'median_income': {'min': 0.5, 'max': 15.0, 'description': 'Median income (tens of thousands)'},
            'median_house_value': {'min': 14999.0, 'max': 500001.0, 'description': 'House value in dollars'}
        }

        for column, expected in expected_ranges.items():
            if column in self.train_data.columns:
                col_data = self.train_data[column]
                actual_min = col_data.min()
                actual_max = col_data.max()

                # Check if values are within expected ranges
                min_valid = actual_min >= expected['min']
                max_valid = actual_max <= expected['max']

                # Count out-of-range values
                out_of_range = (
                    (col_data < expected['min']) | (col_data > expected['max'])
                ).sum()

                validation_results[column] = {
                    'expected_min': expected['min'],
                    'expected_max': expected['max'],
                    'actual_min': float(actual_min),
                    'actual_max': float(actual_max),
                    'min_valid': min_valid,
                    'max_valid': max_valid,
                    'out_of_range_count': int(out_of_range),
                    'out_of_range_percentage': float(out_of_range / len(col_data) * 100),
                    'description': expected['description']
                }

        return validation_results

    def clean_california_housing_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply California housing specific cleaning.

        Returns:
            Tuple of (cleaned_train, cleaned_test)
        """
        print("🏠 Starting California housing data cleaning...")

        # Step 1: Remove duplicates
        self.remove_duplicates()

        # Step 2: Standardize column names
        self.standardize_column_names()

        # Step 3: Validate data ranges
        validation_results = self.validate_data_ranges()

        range_issues = {col: result for col, result in validation_results.items()
                       if not (result['min_valid'] and result['max_valid'])}

        if range_issues:
            print(f"⚠️ Found {len(range_issues)} features with range issues:")
            for col, issue in range_issues.items():
                print(f"  • {col}: Expected [{issue['expected_min']}, {issue['expected_max']}], "
                      f"Got [{issue['actual_min']:.1f}, {issue['actual_max']:.1f}]")
        else:
            print("✅ All feature ranges are valid")

        # Step 4: Handle outliers (conservative approach for California data)
        outlier_columns = ['total_rooms', 'total_bedrooms', 'population', 'households']
        existing_outlier_cols = [col for col in outlier_columns if col in self.train_data.columns]

        if existing_outlier_cols:
            self.handle_outliers(
                columns=existing_outlier_cols,
                method='iqr',
                factor=2.0,  # More conservative than default
                max_outliers_pct=3.0  # Remove at most 3% outliers
            )

        # Step 5: Feature consistency checks
        self._ensure_feature_consistency()

        print(f"✅ California housing data cleaning completed")
        print(f"📊 Final training shape: {self.train_data.shape}")
        if self.test_data is not None:
            print(f"📊 Final test shape: {self.test_data.shape}")

        return self.train_data, self.test_data

    def _ensure_feature_consistency(self) -> None:
        """Ensure logical consistency in California housing features."""
        print("🔧 Ensuring feature consistency...")

        inconsistencies_fixed = 0

        # Ensure total_bedrooms <= total_rooms
        if 'total_bedrooms' in self.train_data.columns and 'total_rooms' in self.train_data.columns:
            invalid_bedrooms = self.train_data['total_bedrooms'] > self.train_data['total_rooms']
            if invalid_bedrooms.any():
                # Cap bedrooms at total rooms
                self.train_data.loc[invalid_bedrooms, 'total_bedrooms'] = \
                    self.train_data.loc[invalid_bedrooms, 'total_rooms']
                inconsistencies_fixed += invalid_bedrooms.sum()

                if self.test_data is not None and 'total_bedrooms' in self.test_data.columns:
                    test_invalid = self.test_data['total_bedrooms'] > self.test_data['total_rooms']
                    if test_invalid.any():
                        self.test_data.loc[test_invalid, 'total_bedrooms'] = \
                            self.test_data.loc[test_invalid, 'total_rooms']

        # Ensure households <= population
        if 'households' in self.train_data.columns and 'population' in self.train_data.columns:
            invalid_households = self.train_data['households'] > self.train_data['population']
            if invalid_households.any():
                # Cap households at population
                self.train_data.loc[invalid_households, 'households'] = \
                    self.train_data.loc[invalid_households, 'population']
                inconsistencies_fixed += invalid_households.sum()

                if self.test_data is not None and 'households' in self.test_data.columns:
                    test_invalid = self.test_data['households'] > self.test_data['population']
                    if test_invalid.any():
                        self.test_data.loc[test_invalid, 'households'] = \
                            self.test_data.loc[test_invalid, 'population']

        if inconsistencies_fixed > 0:
            self.log_operation('ensure_consistency', f'Fixed {inconsistencies_fixed} logical inconsistencies')
            print(f"🔧 Fixed {inconsistencies_fixed} logical inconsistencies")
        else:
            print("✅ No logical inconsistencies found")

    def convert_data_types(self) -> None:
        """Convert data types appropriately for California housing data."""
        type_conversions = []

        # Ensure proper numeric types
        numeric_columns = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value'
        ]

        for col in numeric_columns:
            if col in self.train_data.columns:
                original_dtype = self.train_data[col].dtype

                # Convert to appropriate numeric type
                if col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
                    # Integer counts
                    self.train_data[col] = self.train_data[col].round().astype('int64')
                    if self.test_data is not None and col in self.test_data.columns:
                        self.test_data[col] = self.test_data[col].round().astype('int64')
                    new_dtype = 'int64'
                else:
                    # Float values
                    self.train_data[col] = self.train_data[col].astype('float64')
                    if self.test_data is not None and col in self.test_data.columns:
                        self.test_data[col] = self.test_data[col].astype('float64')
                    new_dtype = 'float64'

                if str(original_dtype) != new_dtype:
                    type_conversions.append(f'{col}: {original_dtype} -> {new_dtype}')

        # Ensure categorical columns are proper strings
        categorical_columns = ['ocean_proximity']
        for col in categorical_columns:
            if col in self.train_data.columns:
                original_dtype = self.train_data[col].dtype
                self.train_data[col] = self.train_data[col].astype('string')
                if self.test_data is not None and col in self.test_data.columns:
                    self.test_data[col] = self.test_data[col].astype('string')

                if str(original_dtype) != 'string':
                    type_conversions.append(f'{col}: {original_dtype} -> string')

        if type_conversions:
            self.log_operation('convert_data_types', f'Converted types: {type_conversions}')
            print(f"🔄 Converted {len(type_conversions)} column types")
        else:
            print("✅ Data types already optimal")

    def add_derived_features(self) -> None:
        """Add useful derived features for California housing."""
        print("🔧 Creating derived features...")

        derived_count = 0

        # Rooms per household
        if 'total_rooms' in self.train_data.columns and 'households' in self.train_data.columns:
            self.train_data['rooms_per_household'] = (
                self.train_data['total_rooms'] / self.train_data['households'].replace(0, 1)
            )
            if self.test_data is not None:
                self.test_data['rooms_per_household'] = (
                    self.test_data['total_rooms'] / self.test_data['households'].replace(0, 1)
                )
            derived_count += 1

        # Bedrooms per room ratio
        if 'total_bedrooms' in self.train_data.columns and 'total_rooms' in self.train_data.columns:
            self.train_data['bedrooms_per_room'] = (
                self.train_data['total_bedrooms'] / self.train_data['total_rooms'].replace(0, 1)
            )
            if self.test_data is not None:
                self.test_data['bedrooms_per_room'] = (
                    self.test_data['total_bedrooms'] / self.test_data['total_rooms'].replace(0, 1)
                )
            derived_count += 1

        # Population per household
        if 'population' in self.train_data.columns and 'households' in self.train_data.columns:
            self.train_data['population_per_household'] = (
                self.train_data['population'] / self.train_data['households'].replace(0, 1)
            )
            if self.test_data is not None:
                self.test_data['population_per_household'] = (
                    self.test_data['population'] / self.test_data['households'].replace(0, 1)
                )
            derived_count += 1

        if derived_count > 0:
            self.log_operation('add_derived_features', f'Added {derived_count} derived features')
            print(f"➕ Added {derived_count} derived features")

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of all cleaning operations.

        Returns:
            Dictionary with cleaning summary
        """
        return {
            'original_train_shape': self.original_train_shape,
            'original_test_shape': self.original_test_shape,
            'final_train_shape': self.train_data.shape,
            'final_test_shape': self.test_data.shape if self.test_data is not None else None,
            'operations_performed': len(self.cleaning_log),
            'cleaning_log': self.cleaning_log,
            'features_added': self.train_data.shape[1] - self.original_train_shape[1],
            'records_removed': self.original_train_shape[0] - self.train_data.shape[0]
        }

    def print_cleaning_summary(self) -> None:
        """Print a summary of cleaning operations."""
        summary = self.get_cleaning_summary()

        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)

        print(f"\nOriginal Data:")
        print(f"  • Training: {summary['original_train_shape']}")
        if summary['original_test_shape']:
            print(f"  • Test: {summary['original_test_shape']}")

        print(f"\nFinal Data:")
        print(f"  • Training: {summary['final_train_shape']}")
        if summary['final_test_shape']:
            print(f"  • Test: {summary['final_test_shape']}")

        print(f"\nChanges:")
        print(f"  • Features added: {summary['features_added']}")
        print(f"  • Records removed: {summary['records_removed']}")
        print(f"  • Operations performed: {summary['operations_performed']}")

        if self.cleaning_log:
            print(f"\nOperations Log:")
            for i, log_entry in enumerate(self.cleaning_log, 1):
                print(f"  {i}. {log_entry['operation']}: {log_entry['details']}")


# Test the module
if __name__ == "__main__":
    print("Testing Data Cleaning module...")

    # Create sample California housing data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Id': range(1, 101),
        'longitude': np.random.uniform(-124, -114, 100),
        'latitude': np.random.uniform(32, 42, 100),
        'housing_median_age': np.random.uniform(1, 52, 100),
        'total_rooms': np.random.uniform(500, 8000, 100),
        'total_bedrooms': np.random.uniform(100, 1500, 100),
        'population': np.random.uniform(300, 5000, 100),
        'households': np.random.uniform(100, 1800, 100),
        'median_income': np.random.uniform(0.5, 15, 100),
        'median_house_value': np.random.uniform(50000, 500000, 100),
        'ocean_proximity': np.random.choice(['NEAR BAY', 'INLAND', 'NEAR OCEAN'], 100)
    })

    # Add some data quality issues for testing
    sample_data.loc[99] = sample_data.loc[98]  # Duplicate
    sample_data.loc[95, 'total_rooms'] = 50000  # Outlier
    sample_data.loc[96, 'total_bedrooms'] = sample_data.loc[96, 'total_rooms'] + 100  # Inconsistency

    try:
        cleaner = DataCleaner(sample_data)
        cleaned_train, _ = cleaner.clean_california_housing_data()
        cleaner.print_cleaning_summary()

        print("\n✅ Data Cleaning module working correctly")
        print(f"Original shape: {sample_data.shape}")
        print(f"Cleaned shape: {cleaned_train.shape}")

    except Exception as e:
        print(f"\n❌ Error in Data Cleaning module: {e}")
        import traceback
        traceback.print_exc()