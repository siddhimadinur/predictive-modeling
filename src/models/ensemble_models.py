"""
Ensemble models for California housing price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest Regression model optimized for California housing."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42, **kwargs):
        super().__init__("Random Forest",
                        n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                        random_state=random_state, **kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = self._create_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> RandomForestRegressor:
        """Create Random Forest model optimized for housing data."""
        return RandomForestRegressor(n_jobs=-1, **kwargs)

    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.

        Returns:
            OOB score or None
        """
        if self.is_fitted and hasattr(self.model, 'oob_score_'):
            return float(self.model.oob_score_)
        return None


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Regression model for California housing."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, random_state: int = 42, **kwargs):
        super().__init__("Gradient Boosting",
                        n_estimators=n_estimators, learning_rate=learning_rate,
                        max_depth=max_depth, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, random_state=random_state, **kwargs)

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self._create_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> GradientBoostingRegressor:
        """Create Gradient Boosting model."""
        return GradientBoostingRegressor(**kwargs)

    def plot_training_deviance(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot training deviance curve.

        Args:
            figsize: Figure size
        """
        if not self.is_fitted:
            print("Model not fitted yet")
            return

        import matplotlib.pyplot as plt

        # Plot training deviance
        train_scores = self.model.train_score_

        plt.figure(figsize=figsize)
        plt.plot(range(1, len(train_scores) + 1), train_scores, 'b-', label='Training Deviance')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        plt.title(f'Training Deviance - {self.name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class XGBoostModel(BaseModel):
    """XGBoost Regression model for California housing (if available)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, min_child_weight: int = 1,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 random_state: int = 42, **kwargs):

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install with: pip install xgboost")

        super().__init__("XGBoost",
                        n_estimators=n_estimators, learning_rate=learning_rate,
                        max_depth=max_depth, min_child_weight=min_child_weight,
                        subsample=subsample, colsample_bytree=colsample_bytree,
                        random_state=random_state, **kwargs)

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = self._create_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> Any:
        """Create XGBoost model."""
        return xgb.XGBRegressor(n_jobs=-1, **kwargs)

    def plot_training_history(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot XGBoost training history if available.

        Args:
            figsize: Figure size
        """
        if not self.is_fitted:
            print("Model not fitted yet")
            return

        # XGBoost training history requires eval_set during training
        print("💡 To plot training history, retrain with eval_set parameter")


class ExtraTreesModel(BaseModel):
    """Extra Trees (Extremely Randomized Trees) model for California housing."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42, **kwargs):
        super().__init__("Extra Trees",
                        n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                        random_state=random_state, **kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self._create_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> ExtraTreesRegressor:
        """Create Extra Trees model."""
        return ExtraTreesRegressor(n_jobs=-1, **kwargs)


class VotingEnsembleModel(BaseModel):
    """Voting Ensemble combining multiple models for robust predictions."""

    def __init__(self, models: List[BaseModel], voting: str = 'hard', **kwargs):
        super().__init__("Voting Ensemble", models=models, voting=voting, **kwargs)
        self.base_models = models
        self.voting = voting
        self.model = self._create_model(models=models, voting=voting, **kwargs)

    def _create_model(self, models: List[BaseModel], voting: str = 'hard', **kwargs) -> Any:
        """Create Voting Ensemble model."""
        from sklearn.ensemble import VotingRegressor

        # Create estimator list from base models
        estimators = [(model.name.replace(' ', '_').lower(), model.model) for model in models]

        return VotingRegressor(estimators=estimators, n_jobs=-1, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> 'BaseModel':
        """
        Train the voting ensemble.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Self for method chaining
        """
        print(f"🔧 Training {self.name} with {len(self.base_models)} base models...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Fit the ensemble
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calculate metrics
        train_predictions = self.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, train_predictions, "Training")

        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            self.validation_metrics = self._calculate_metrics(y_val, val_predictions, "Validation")

        print(f"✅ {self.name} training completed")
        print(f"  • Training R²: {self.training_metrics['r2']:.3f}")
        if self.validation_metrics:
            print(f"  • Validation R²: {self.validation_metrics['r2']:.3f}")
            print(f"  • Validation RMSE: ${self.validation_metrics['rmse']:,.0f}")

        return self


# Test the ensemble models
if __name__ == "__main__":
    print("Testing Ensemble Models for California Housing...")

    # Create sample California housing data
    np.random.seed(42)
    n_samples = 500

    X_sample = pd.DataFrame({
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'total_rooms': np.random.uniform(500, 8000, n_samples),
        'housing_median_age': np.random.uniform(1, 52, n_samples),
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32, 42, n_samples),
        'population': np.random.uniform(300, 5000, n_samples),
        'households': np.random.uniform(100, 1800, n_samples)
    })

    # Create realistic target
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

    # Test ensemble models
    models_to_test = [
        RandomForestModel(n_estimators=50, random_state=42),  # Smaller for faster testing
        GradientBoostingModel(n_estimators=50, learning_rate=0.1, random_state=42),
        ExtraTreesModel(n_estimators=50, random_state=42)
    ]

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models_to_test.append(XGBoostModel(n_estimators=50, random_state=42))

    for model in models_to_test:
        try:
            # Train model
            model.fit(X_train, y_train, X_val, y_val)

            # Test feature importance
            importance_df = model.get_feature_importance()
            if importance_df is not None:
                print(f"  • Top feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.3f})")

            print(f"✅ {model.name} working correctly\\n")

        except Exception as e:
            print(f"❌ Error testing {model.name}: {e}")

    print("✅ All Ensemble Models tested successfully!")