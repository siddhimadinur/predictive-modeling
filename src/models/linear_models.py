"""
Linear regression models for California housing price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """Standard Linear Regression model for California housing."""

    def __init__(self, **kwargs):
        super().__init__("Linear Regression", **kwargs)
        self.model = self._create_model(**kwargs)

    def _create_model(self, **kwargs) -> LinearRegression:
        """Create Linear Regression model."""
        return LinearRegression(**kwargs)


class RidgeRegressionModel(BaseModel):
    """Ridge Regression model with L2 regularization for California housing."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Ridge Regression", alpha=alpha, **kwargs)
        self.alpha = alpha
        self.model = self._create_model(alpha=alpha, **kwargs)

    def _create_model(self, alpha: float = 1.0, **kwargs) -> Ridge:
        """Create Ridge Regression model."""
        return Ridge(alpha=alpha, random_state=42, **kwargs)


class LassoRegressionModel(BaseModel):
    """Lasso Regression model with L1 regularization for feature selection."""

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__("Lasso Regression", alpha=alpha, **kwargs)
        self.alpha = alpha
        self.model = self._create_model(alpha=alpha, **kwargs)

    def _create_model(self, alpha: float = 0.1, **kwargs) -> Lasso:
        """Create Lasso Regression model."""
        return Lasso(alpha=alpha, random_state=42, max_iter=2000, **kwargs)

    def get_selected_features(self) -> List[str]:
        """
        Get features selected by Lasso (non-zero coefficients).

        Returns:
            List of selected feature names
        """
        if not self.is_fitted or not hasattr(self.model, 'coef_'):
            return []

        selected_indices = np.where(self.model.coef_ != 0)[0]
        return [self.feature_names[i] for i in selected_indices]

    def print_feature_selection_summary(self) -> None:
        """Print summary of Lasso feature selection."""
        if not self.is_fitted:
            print("Model not fitted yet")
            return

        selected_features = self.get_selected_features()
        total_features = len(self.feature_names)

        print(f"\n🎯 LASSO FEATURE SELECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total features: {total_features}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Eliminated features: {total_features - len(selected_features)}")
        print(f"Selection rate: {len(selected_features)/total_features*100:.1f}%")

        if selected_features:
            print(f"\nSelected Features:")
            for i, feature in enumerate(selected_features, 1):
                coef_value = self.model.coef_[self.feature_names.index(feature)]
                print(f"  {i:2d}. {feature:<25}: {coef_value:+.3f}")


class ElasticNetModel(BaseModel):
    """ElasticNet Regression combining L1 and L2 regularization."""

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, **kwargs):
        super().__init__("ElasticNet Regression", alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = self._create_model(alpha=alpha, l1_ratio=l1_ratio, **kwargs)

    def _create_model(self, alpha: float = 0.1, l1_ratio: float = 0.5, **kwargs) -> ElasticNet:
        """Create ElasticNet Regression model."""
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000, **kwargs)


class PolynomialRegressionModel(BaseModel):
    """Polynomial Regression model for capturing non-linear relationships."""

    def __init__(self, degree: int = 2, alpha: float = 1.0, **kwargs):
        super().__init__("Polynomial Regression", degree=degree, alpha=alpha, **kwargs)
        self.degree = degree
        self.alpha = alpha
        self.model = self._create_model(degree=degree, alpha=alpha, **kwargs)

    def _create_model(self, degree: int = 2, alpha: float = 1.0, **kwargs) -> Pipeline:
        """Create Polynomial Regression pipeline."""
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])

    def get_polynomial_feature_names(self, input_features: List[str]) -> List[str]:
        """
        Get names of polynomial features created.

        Args:
            input_features: Original feature names

        Returns:
            List of polynomial feature names
        """
        if not self.is_fitted:
            return []

        poly_transformer = self.model.named_steps['poly']
        return poly_transformer.get_feature_names_out(input_features)


# Test the linear models
if __name__ == "__main__":
    print("Testing Linear Models for California Housing...")

    # Create sample California housing data
    np.random.seed(42)
    n_samples = 200

    X_sample = pd.DataFrame({
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'total_rooms': np.random.uniform(500, 8000, n_samples),
        'housing_median_age': np.random.uniform(1, 52, n_samples),
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32, 42, n_samples)
    })

    # Create realistic target with noise
    y_sample = (
        X_sample['median_income'] * 40000 +
        X_sample['total_rooms'] * 30 +
        (50 - X_sample['housing_median_age']) * 1000 +
        np.random.normal(0, 20000, n_samples)
    )
    y_sample = pd.Series(np.clip(y_sample, 50000, 500000), name='median_house_value')

    # Split for testing
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_sample[:split_idx], X_sample[split_idx:]
    y_train, y_val = y_sample[:split_idx], y_sample[split_idx:]

    print(f"\n🧪 Testing with {len(X_train)} training samples...")

    # Test each linear model
    models_to_test = [
        LinearRegressionModel(),
        RidgeRegressionModel(alpha=1.0),
        LassoRegressionModel(alpha=0.1),
        ElasticNetModel(alpha=0.1, l1_ratio=0.5),
        PolynomialRegressionModel(degree=2, alpha=1.0)
    ]

    for model in models_to_test:
        try:
            # Train model
            model.fit(X_train, y_train, X_val, y_val)

            # Print performance summary
            model.print_performance_summary()

            # Test special methods
            if isinstance(model, LassoRegressionModel):
                model.print_feature_selection_summary()

            print(f"✅ {model.name} working correctly\\n")

        except Exception as e:
            print(f"❌ Error testing {model.name}: {e}")

    print("✅ All Linear Models tested successfully!")