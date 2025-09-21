"""
Complete data preprocessing pipeline for California housing price prediction.
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
from src.feature_engineering import CaliforniaHousingFeatureEngineer
from src.utils import save_data, ensure_dir_exists


class CaliforniaHousingPipeline:
    """Complete data preprocessing pipeline for California housing data."""

    def __init__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None,
                 target_col: str = 'median_house_value'):
        """
        Initialize preprocessing pipeline.

        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
            target_col: Target column name
        """
        self.original_train = train_data.copy()
        self.original_test = test_data.copy() if test_data is not None else None

        # Auto-detect target column if not found
        target_candidates = ['median_house_value', 'SalePrice', 'price', 'target']
        self.target_col = target_col

        if self.target_col not in train_data.columns:
            for candidate in target_candidates:
                if candidate in train_data.columns:
                    self.target_col = candidate
                    print(f"🎯 Auto-detected target column: '{self.target_col}'")
                    break

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
            'outlier_factor': 2.0,  # Conservative for housing data
            'max_outliers_pct': 3.0,
            'encoding_method': 'onehot',
            'scaling_method': 'robust',
            'include_interactions': True,
            'include_polynomials': True,
            'polynomial_degree': 2,
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

    def clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean the data using California housing specific methods.

        Returns:
            Tuple of (cleaned_train, cleaned_test)
        """
        print("🧹 Step 1: Data Cleaning")
        print("-" * 30)

        self.cleaner = DataCleaner(self.original_train, self.original_test)

        # Apply California housing specific cleaning
        train_clean, test_clean = self.cleaner.clean_california_housing_data()

        self.log_step('clean_data', 'Applied California housing data cleaning', train_clean.shape)

        return train_clean, test_clean

    def handle_missing_values(self, train_data: pd.DataFrame,
                            test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values (California housing data is typically complete).

        Args:
            train_data: Training data
            test_data: Test data

        Returns:
            Tuple of (imputed_train, imputed_test)
        """
        print("\n🔧 Step 2: Missing Value Handling")
        print("-" * 30)

        self.missing_handler = MissingValueHandler(train_data, test_data)

        # Check if there are any missing values
        missing_analysis = self.missing_handler.analyze_missing_patterns()

        if missing_analysis['total_missing'] == 0:
            print("✅ No missing values found - data is complete!")
            self.log_step('handle_missing', 'No missing values found', train_data.shape)
            return train_data, test_data

        print(f"⚠️ Found {missing_analysis['total_missing']} missing values")

        # Create and apply imputation strategy
        strategies = self.missing_handler.create_housing_imputation_strategy(missing_analysis)
        train_imputed, test_imputed = self.missing_handler.apply_imputation(strategies)

        # Validate imputation
        validation_results = self.missing_handler.validate_imputation(train_imputed, test_imputed)
        self.log_step('handle_missing', f'Applied {len(strategies)} imputation strategies', train_imputed.shape)

        print(f"✅ Missing value handling completed")

        return train_imputed, test_imputed

    def engineer_features(self, train_data: pd.DataFrame,
                         test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Engineer features specific to California housing.

        Args:
            train_data: Training data
            test_data: Test data

        Returns:
            Tuple of (engineered_train, engineered_test)
        """
        print("\n🏗️ Step 3: Feature Engineering")
        print("-" * 30)

        self.feature_engineer = CaliforniaHousingFeatureEngineer(train_data, test_data, self.target_col)

        train_engineered, test_engineered = self.feature_engineer.apply_feature_engineering_pipeline(
            include_interactions=self.pipeline_config['include_interactions'],
            include_polynomials=self.pipeline_config['include_polynomials'],
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
        print(f"\n📊 Step 4: Data Splitting")
        print("-" * 30)

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

        print(f"✅ Data split completed:")
        print(f"  • Training: {X_train.shape}")
        print(f"  • Validation: {X_val.shape}")

        return X_train, X_val, y_train, y_val

    def run_pipeline(self, save_processed: bool = True) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.

        Args:
            save_processed: Whether to save processed data

        Returns:
            Dictionary with pipeline results
        """
        print("🚀 CALIFORNIA HOUSING DATA PREPROCESSING PIPELINE")
        print("="*60)

        # Step 1: Clean data
        train_clean, test_clean = self.clean_data()

        # Step 2: Handle missing values
        train_imputed, test_imputed = self.handle_missing_values(train_clean, test_clean)

        # Step 3: Engineer features
        train_engineered, test_engineered = self.engineer_features(train_imputed, test_imputed)

        # For test data, remove target column if it exists (for California housing split from single CSV)
        if test_engineered is not None and self.target_col in test_engineered.columns:
            test_engineered = test_engineered.drop(self.target_col, axis=1)
            print(f"🎯 Removed target column from test data for prediction setup")

        # Step 4: Split data
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
            'pipeline_config': self.pipeline_config,
            'target_column': self.target_col
        }

        # Print final summary
        print("\n" + "="*60)
        print("PIPELINE RESULTS SUMMARY")
        print("="*60)
        print(f"Original data: {results['original_train_shape']}")
        print(f"Final features: {results['feature_count']}")
        print(f"Training samples: {results['X_train_shape'][0]}")
        print(f"Validation samples: {results['X_val_shape'][0]}")
        print(f"Target column: {results['target_column']}")
        print("✅ Pipeline completed successfully!")

        return results

    def _save_processed_data(self) -> None:
        """Save processed data to files."""
        print(f"\n💾 Saving processed data...")
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
            save_data(self.y_train.to_frame(self.target_col), PROCESSED_DATA_DIR / 'y_train.csv')
            save_data(self.y_val.to_frame(self.target_col), PROCESSED_DATA_DIR / 'y_val.csv')

        # Save pipeline components and artifacts
        pipeline_artifacts = {
            'cleaner': self.cleaner,
            'missing_handler': self.missing_handler,
            'feature_engineer': self.feature_engineer,
            'pipeline_config': self.pipeline_config,
            'processing_log': self.processing_log,
            'target_column': self.target_col,
            'feature_names': list(self.X_train.columns) if self.X_train is not None else []
        }

        joblib.dump(pipeline_artifacts, PROCESSED_DATA_DIR / 'pipeline_artifacts.pkl')

        print(f"✅ Processed data saved to: {PROCESSED_DATA_DIR}")

    def get_processing_summary(self) -> str:
        """
        Get a text summary of the processing pipeline.

        Returns:
            Processing summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("CALIFORNIA HOUSING DATA PREPROCESSING SUMMARY")
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
            summary.append(f"Target column: {self.target_col}")
            summary.append(f"Feature count: {self.X_train.shape[1]}")
            summary.append(f"Training samples: {self.X_train.shape[0]}")
            summary.append(f"Validation samples: {self.X_val.shape[0]}")

            # Feature engineering summary
            if self.feature_engineer:
                feature_summary = self.feature_engineer.get_feature_summary()
                summary.append(f"\nFEATURE ENGINEERING:")
                summary.append(f"Original features: {feature_summary['original_features']}")
                summary.append(f"Created features: {feature_summary['total_created']}")
                summary.append(f"Final feature count: {feature_summary['final_feature_count']}")

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

        # Load pipeline artifacts
        artifacts_path = data_dir / 'pipeline_artifacts.pkl'
        if artifacts_path.exists():
            try:
                artifacts = joblib.load(artifacts_path)
                datasets['pipeline_artifacts'] = artifacts
            except Exception as e:
                print(f"⚠️ Could not load pipeline artifacts: {e}")

        return datasets

    def predict_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the preprocessing pipeline to new data for prediction.

        Args:
            new_data: New data to preprocess

        Returns:
            Preprocessed data ready for model prediction
        """
        if not all(x is not None for x in [self.cleaner, self.feature_engineer]):
            raise ValueError("Pipeline has not been fitted yet. Run run_pipeline() first.")

        print("🔄 Applying preprocessing pipeline to new data...")

        # Apply same transformations
        processed_data = new_data.copy()

        # Apply feature engineering transformations
        if self.feature_engineer:
            # This would require implementing transform methods in the feature engineer
            # For now, return a warning
            print("⚠️ Transform method for new data not yet implemented")

        return processed_data

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted pipeline.

        Returns:
            Pipeline information dictionary
        """
        return {
            'is_fitted': all(x is not None for x in [self.X_train, self.y_train]),
            'target_column': self.target_col,
            'feature_count': self.X_train.shape[1] if self.X_train is not None else 0,
            'training_samples': len(self.X_train) if self.X_train is not None else 0,
            'validation_samples': len(self.X_val) if self.X_val is not None else 0,
            'pipeline_config': self.pipeline_config,
            'processing_steps': len(self.processing_log),
            'created_features': len(self.feature_engineer.created_features) if self.feature_engineer else 0
        }


# Test the complete pipeline
if __name__ == "__main__":
    print("Testing California Housing Preprocessing Pipeline...")

    # Create sample California housing data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Id': range(1, 201),
        'longitude': np.random.uniform(-124, -114, 200),
        'latitude': np.random.uniform(32, 42, 200),
        'housing_median_age': np.random.uniform(1, 52, 200),
        'total_rooms': np.random.uniform(500, 8000, 200),
        'total_bedrooms': np.random.uniform(100, 1500, 200),
        'population': np.random.uniform(300, 5000, 200),
        'households': np.random.uniform(100, 1800, 200),
        'median_income': np.random.uniform(0.5, 15, 200),
        'median_house_value': np.random.uniform(50000, 500000, 200),
        'ocean_proximity': np.random.choice(['NEAR BAY', 'INLAND', 'NEAR OCEAN'], 200)
    })

    # Create test data (without target for real scenario)
    test_data = sample_data.drop('median_house_value', axis=1).sample(50).reset_index(drop=True)

    try:
        # Run complete pipeline
        pipeline = CaliforniaHousingPipeline(sample_data, test_data)
        results = pipeline.run_pipeline(save_processed=False)

        # Print results
        print(f"\n✅ Complete pipeline working correctly!")
        print(f"Features engineered: {results['feature_count']}")
        print(f"Training samples: {results['X_train_shape'][0]}")
        print(f"Validation samples: {results['X_val_shape'][0]}")

        # Show feature engineering summary
        if pipeline.feature_engineer:
            pipeline.feature_engineer.print_feature_summary()

    except Exception as e:
        print(f"\n❌ Error in complete pipeline: {e}")
        import traceback
        traceback.print_exc()