"""
Feature engineering utilities optimized for California housing data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression


class CaliforniaHousingFeatureEngineer:
    """Feature engineering class optimized for California housing data."""

    def __init__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None,
                 target_col: str = 'median_house_value'):
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

    def create_geographic_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create geographic-based features for California housing.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        print("🗺️ Creating geographic features...")

        # Distance from major California cities
        if 'longitude' in train_enhanced.columns and 'latitude' in train_enhanced.columns:
            # Major California city coordinates
            cities = {
                'Los_Angeles': {'lat': 34.0522, 'lon': -118.2437},
                'San_Francisco': {'lat': 37.7749, 'lon': -122.4194},
                'San_Diego': {'lat': 32.7157, 'lon': -117.1611},
                'Sacramento': {'lat': 38.5816, 'lon': -121.4944}
            }

            for city_name, coords in cities.items():
                # Calculate distance (approximate using Euclidean distance)
                train_enhanced[f'distance_to_{city_name}'] = np.sqrt(
                    (train_enhanced['latitude'] - coords['lat'])**2 +
                    (train_enhanced['longitude'] - coords['lon'])**2
                )
                self.created_features.append(f'distance_to_{city_name}')

                if test_enhanced is not None:
                    test_enhanced[f'distance_to_{city_name}'] = np.sqrt(
                        (test_enhanced['latitude'] - coords['lat'])**2 +
                        (test_enhanced['longitude'] - coords['lon'])**2
                    )

            # Create geographic clusters based on location
            # Northern vs Southern California
            train_enhanced['is_northern_ca'] = (train_enhanced['latitude'] > 36.0).astype(int)
            if test_enhanced is not None:
                test_enhanced['is_northern_ca'] = (test_enhanced['latitude'] > 36.0).astype(int)
            self.created_features.append('is_northern_ca')

            # Coastal vs Inland (rough approximation)
            train_enhanced['is_coastal'] = (train_enhanced['longitude'] > -121.0).astype(int)
            if test_enhanced is not None:
                test_enhanced['is_coastal'] = (test_enhanced['longitude'] > -121.0).astype(int)
            self.created_features.append('is_coastal')

        return train_enhanced, test_enhanced

    def create_housing_density_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create housing density and ratio features.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        print("🏘️ Creating housing density features...")

        # Rooms per household
        if 'total_rooms' in train_enhanced.columns and 'households' in train_enhanced.columns:
            train_enhanced['rooms_per_household'] = (
                train_enhanced['total_rooms'] / train_enhanced['households'].replace(0, 1)
            )
            if test_enhanced is not None:
                test_enhanced['rooms_per_household'] = (
                    test_enhanced['total_rooms'] / test_enhanced['households'].replace(0, 1)
                )
            self.created_features.append('rooms_per_household')

        # Bedrooms per room ratio
        if 'total_bedrooms' in train_enhanced.columns and 'total_rooms' in train_enhanced.columns:
            train_enhanced['bedrooms_per_room'] = (
                train_enhanced['total_bedrooms'] / train_enhanced['total_rooms'].replace(0, 1)
            )
            if test_enhanced is not None:
                test_enhanced['bedrooms_per_room'] = (
                    test_enhanced['total_bedrooms'] / test_enhanced['total_rooms'].replace(0, 1)
                )
            self.created_features.append('bedrooms_per_room')

        # Population per household
        if 'population' in train_enhanced.columns and 'households' in train_enhanced.columns:
            train_enhanced['population_per_household'] = (
                train_enhanced['population'] / train_enhanced['households'].replace(0, 1)
            )
            if test_enhanced is not None:
                test_enhanced['population_per_household'] = (
                    test_enhanced['population'] / test_enhanced['households'].replace(0, 1)
                )
            self.created_features.append('population_per_household')

        # Population density (population per total rooms)
        if 'population' in train_enhanced.columns and 'total_rooms' in train_enhanced.columns:
            train_enhanced['population_density'] = (
                train_enhanced['population'] / train_enhanced['total_rooms'].replace(0, 1)
            )
            if test_enhanced is not None:
                test_enhanced['population_density'] = (
                    test_enhanced['population'] / test_enhanced['total_rooms'].replace(0, 1)
                )
            self.created_features.append('population_density')

        return train_enhanced, test_enhanced

    def create_income_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create income-related features.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        print("💰 Creating income-based features...")

        if 'median_income' in train_enhanced.columns:
            # Income categories
            income_bins = [0, 2, 4, 6, 8, 10, 15]
            income_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High', 'Ultra_High']

            train_enhanced['income_category'] = pd.cut(
                train_enhanced['median_income'],
                bins=income_bins,
                labels=income_labels,
                include_lowest=True
            ).astype(str)

            if test_enhanced is not None:
                test_enhanced['income_category'] = pd.cut(
                    test_enhanced['median_income'],
                    bins=income_bins,
                    labels=income_labels,
                    include_lowest=True
                ).astype(str)

            self.created_features.append('income_category')

            # Log-transformed income (since income often has exponential relationship with price)
            train_enhanced['log_median_income'] = np.log1p(train_enhanced['median_income'])
            if test_enhanced is not None:
                test_enhanced['log_median_income'] = np.log1p(test_enhanced['median_income'])
            self.created_features.append('log_median_income')

            # Income squared (capture non-linear relationships)
            train_enhanced['median_income_squared'] = train_enhanced['median_income'] ** 2
            if test_enhanced is not None:
                test_enhanced['median_income_squared'] = test_enhanced['median_income'] ** 2
            self.created_features.append('median_income_squared')

        return train_enhanced, test_enhanced

    def create_age_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create housing age-related features.

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        print("🏠 Creating age-based features...")

        if 'housing_median_age' in train_enhanced.columns:
            # Age categories
            age_bins = [0, 10, 20, 30, 40, 52]
            age_labels = ['New', 'Modern', 'Mature', 'Older', 'Vintage']

            train_enhanced['age_category'] = pd.cut(
                train_enhanced['housing_median_age'],
                bins=age_bins,
                labels=age_labels,
                include_lowest=True
            ).astype(str)

            if test_enhanced is not None:
                test_enhanced['age_category'] = pd.cut(
                    test_enhanced['housing_median_age'],
                    bins=age_bins,
                    labels=age_labels,
                    include_lowest=True
                ).astype(str)

            self.created_features.append('age_category')

            # Is new construction (less than 10 years old)
            train_enhanced['is_new_construction'] = (train_enhanced['housing_median_age'] < 10).astype(int)
            if test_enhanced is not None:
                test_enhanced['is_new_construction'] = (test_enhanced['housing_median_age'] < 10).astype(int)
            self.created_features.append('is_new_construction')

        return train_enhanced, test_enhanced

    def create_interaction_features(self, important_features: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create interaction features for California housing.

        Args:
            important_features: List of important features for interactions

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        print("🔗 Creating interaction features...")

        if important_features is None:
            # Default important features for California housing
            important_features = ['median_income', 'total_rooms', 'households', 'housing_median_age']

        # Filter to existing features
        existing_features = [f for f in important_features if f in train_enhanced.columns]
        numerical_features = [f for f in existing_features if train_enhanced[f].dtype in ['int64', 'float64']]

        # Create meaningful interactions for housing data
        if 'median_income' in numerical_features and 'total_rooms' in numerical_features:
            # Income × rooms interaction (affluent areas with large homes)
            train_enhanced['income_rooms_interaction'] = (
                train_enhanced['median_income'] * train_enhanced['total_rooms'] / 1000
            )
            if test_enhanced is not None:
                test_enhanced['income_rooms_interaction'] = (
                    test_enhanced['median_income'] * test_enhanced['total_rooms'] / 1000
                )
            self.created_features.append('income_rooms_interaction')

        if 'median_income' in numerical_features and 'housing_median_age' in numerical_features:
            # Income × age interaction (newer homes in affluent areas)
            train_enhanced['income_age_interaction'] = (
                train_enhanced['median_income'] * (50 - train_enhanced['housing_median_age'])
            )
            if test_enhanced is not None:
                test_enhanced['income_age_interaction'] = (
                    test_enhanced['median_income'] * (50 - test_enhanced['housing_median_age'])
                )
            self.created_features.append('income_age_interaction')

        return train_enhanced, test_enhanced

    def encode_categorical_features(self, encoding_method: str = 'onehot') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features.

        Args:
            encoding_method: 'onehot' or 'label'

        Returns:
            Tuple of (encoded_train, encoded_test)
        """
        train_encoded = self.train_data.copy()
        test_encoded = self.test_data.copy() if self.test_data is not None else None

        print(f"🏷️ Encoding categorical features using {encoding_method}...")

        # Get categorical columns
        categorical_columns = train_encoded.select_dtypes(include=['object', 'string']).columns.tolist()

        for column in categorical_columns:
            if column in ['Id']:  # Skip ID columns
                continue

            unique_count = train_encoded[column].nunique()

            if encoding_method == 'onehot':
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                encoded_features = encoder.fit_transform(train_encoded[[column]])

                # Create feature names
                categories = encoder.categories_[0]
                if len(categories) > 1:  # Only if we have features after dropping first
                    feature_names = [f'{column}_{cat}' for cat in categories[1:]]  # Skip first (dropped)

                    # Add encoded features to dataframe
                    if len(feature_names) > 0:
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

                        print(f"  ✅ One-hot encoded {column}: {len(feature_names)} new features")

                # Remove original categorical column
                train_encoded = train_encoded.drop(column, axis=1)
                if test_encoded is not None:
                    test_encoded = test_encoded.drop(column, axis=1)

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

                print(f"  ✅ Label encoded {column}")

                # Remove original categorical column
                train_encoded = train_encoded.drop(column, axis=1)
                if test_encoded is not None:
                    test_encoded = test_encoded.drop(column, axis=1)

        return train_encoded, test_encoded

    def create_polynomial_features(self, features: List[str] = None, degree: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create polynomial features for key variables.

        Args:
            features: Features to create polynomial terms for
            degree: Polynomial degree

        Returns:
            Tuple of (enhanced_train, enhanced_test)
        """
        train_enhanced = self.train_data.copy()
        test_enhanced = self.test_data.copy() if self.test_data is not None else None

        if features is None:
            # Key features that might benefit from polynomial terms
            features = ['median_income', 'total_rooms', 'housing_median_age']

        existing_features = [f for f in features if f in train_enhanced.columns and train_enhanced[f].dtype in ['int64', 'float64']]

        if existing_features:
            print(f"📈 Creating polynomial features (degree {degree})...")

            from sklearn.preprocessing import PolynomialFeatures

            for feature in existing_features:
                if degree == 2:
                    # Add squared term
                    train_enhanced[f'{feature}_squared'] = train_enhanced[feature] ** 2
                    if test_enhanced is not None:
                        test_enhanced[f'{feature}_squared'] = test_enhanced[feature] ** 2
                    self.created_features.append(f'{feature}_squared')

                elif degree == 3:
                    # Add squared and cubed terms
                    train_enhanced[f'{feature}_squared'] = train_enhanced[feature] ** 2
                    train_enhanced[f'{feature}_cubed'] = train_enhanced[feature] ** 3
                    if test_enhanced is not None:
                        test_enhanced[f'{feature}_squared'] = test_enhanced[feature] ** 2
                        test_enhanced[f'{feature}_cubed'] = test_enhanced[feature] ** 3
                    self.created_features.extend([f'{feature}_squared', f'{feature}_cubed'])

            print(f"  ✅ Created polynomial features for {len(existing_features)} variables")

        return train_enhanced, test_enhanced

    def scale_features(self, numerical_features: List[str] = None,
                      scaling_method: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            # Remove target and ID columns from scaling
            numerical_features = [col for col in numerical_features
                                if col not in [self.target_col, 'Id'] and not col.startswith('is_')]

        if len(numerical_features) == 0:
            print("⚠️ No numerical features found for scaling")
            return train_scaled, test_scaled

        print(f"⚖️ Scaling {len(numerical_features)} features using {scaling_method} scaler...")

        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:  # minmax
            scaler = MinMaxScaler()

        # Fit scaler on training data and transform both train and test
        train_scaled[numerical_features] = scaler.fit_transform(train_scaled[numerical_features])
        self.feature_transformers['scaler'] = scaler

        if test_scaled is not None:
            test_scaled[numerical_features] = scaler.transform(test_scaled[numerical_features])

        print(f"  ✅ Scaled features: {numerical_features[:5]}{'...' if len(numerical_features) > 5 else ''}")

        return train_scaled, test_scaled

    def apply_feature_engineering_pipeline(self, include_interactions: bool = True,
                                         include_polynomials: bool = True,
                                         encoding_method: str = 'onehot',
                                         scaling_method: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply complete feature engineering pipeline for California housing.

        Args:
            include_interactions: Whether to create interaction features
            include_polynomials: Whether to create polynomial features
            encoding_method: Categorical encoding method
            scaling_method: Numerical scaling method

        Returns:
            Tuple of (engineered_train, engineered_test)
        """
        print("🔧 Starting California Housing Feature Engineering Pipeline...")
        print("="*60)

        # Start with copies of original data
        self.train_data = self.train_data.copy()
        self.test_data = self.test_data.copy() if self.test_data is not None else None

        # Step 1: Create geographic features
        self.train_data, self.test_data = self.create_geographic_features()

        # Step 2: Create housing density features
        self.train_data, self.test_data = self.create_housing_density_features()

        # Step 3: Create income features
        self.train_data, self.test_data = self.create_income_features()

        # Step 4: Create age features
        self.train_data, self.test_data = self.create_age_features()

        # Step 5: Create interaction features
        if include_interactions:
            self.train_data, self.test_data = self.create_interaction_features()

        # Step 6: Create polynomial features
        if include_polynomials:
            self.train_data, self.test_data = self.create_polynomial_features()

        # Step 7: Encode categorical features
        self.train_data, self.test_data = self.encode_categorical_features(encoding_method)

        # Step 8: Scale numerical features
        self.train_data, self.test_data = self.scale_features(scaling_method=scaling_method)

        print("="*60)
        print("✅ Feature engineering pipeline completed!")

        return self.train_data, self.test_data

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature engineering operations.

        Returns:
            Feature engineering summary
        """
        original_features = self.train_data.shape[1] - len(self.created_features)

        return {
            'original_features': original_features,
            'created_features': self.created_features,
            'total_created': len(self.created_features),
            'final_feature_count': self.train_data.shape[1],
            'transformers_fitted': list(self.feature_transformers.keys()),
            'feature_categories': {
                'geographic': [f for f in self.created_features if 'distance' in f or 'northern' in f or 'coastal' in f],
                'density': [f for f in self.created_features if 'per_' in f or 'density' in f],
                'income': [f for f in self.created_features if 'income' in f],
                'age': [f for f in self.created_features if 'age' in f or 'construction' in f],
                'polynomial': [f for f in self.created_features if 'squared' in f or 'cubed' in f],
                'interaction': [f for f in self.created_features if 'interaction' in f],
                'encoded': [f for f in self.created_features if any(cat in f for cat in ['NEAR', 'INLAND', 'OCEAN'])]
            }
        }

    def print_feature_summary(self) -> None:
        """Print a detailed summary of feature engineering."""
        summary = self.get_feature_summary()

        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)

        print(f"\nFeature Count:")
        print(f"  • Original features: {summary['original_features']}")
        print(f"  • Created features: {summary['total_created']}")
        print(f"  • Final feature count: {summary['final_feature_count']}")

        print(f"\nFeatures by Category:")
        for category, features in summary['feature_categories'].items():
            if features:
                print(f"  • {category.title()}: {len(features)} features")
                for feature in features[:3]:
                    print(f"    - {feature}")
                if len(features) > 3:
                    print(f"    - ... and {len(features) - 3} more")

        print(f"\nTransformers Fitted:")
        for transformer in summary['transformers_fitted']:
            print(f"  • {transformer}")


# Test the module
if __name__ == "__main__":
    print("Testing California Housing Feature Engineering...")

    # Create sample California housing data
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

    try:
        engineer = CaliforniaHousingFeatureEngineer(sample_data)
        train_engineered, _ = engineer.apply_feature_engineering_pipeline()
        engineer.print_feature_summary()

        print(f"\n✅ Feature Engineering module working correctly")
        print(f"Original shape: {sample_data.shape}")
        print(f"Engineered shape: {train_engineered.shape}")

    except Exception as e:
        print(f"\n❌ Error in Feature Engineering module: {e}")
        import traceback
        traceback.print_exc()