# Phase 4: Model Development & Training

## Overview
Develop, train, and evaluate multiple machine learning models for house price prediction. Implement proper model validation, hyperparameter tuning, and model comparison to select the best performing model.

## Objectives
- Implement multiple regression algorithms
- Perform cross-validation and hyperparameter tuning
- Evaluate model performance with appropriate metrics
- Compare models and select the best performer
- Create model training pipeline
- Save trained models for deployment
- Generate model performance reports

## Step-by-Step Implementation

### 4.1 Model Training Infrastructure

#### 4.1.1 Create Base Model Class
Create `src/models/base_model.py`:
```python
"""
Base model class for house price prediction.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import joblib
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, name: str, **kwargs):
        """
        Initialize base model.

        Args:
            name: Model name
            **kwargs: Model parameters
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
        self.validation_metrics = {}
        self.cv_scores = {}

    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """
        Create the underlying model instance.

        Returns:
            Model instance
        """
        pass

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> 'BaseModel':
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Self for method chaining
        """
        # Store feature names
        self.feature_names = list(X_train.columns)

        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calculate training metrics
        train_predictions = self.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, train_predictions, "Training")

        # Calculate validation metrics if validation set provided
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            self.validation_metrics = self._calculate_metrics(y_val, val_predictions, "Validation")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features for prediction

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
                      scoring: List[str] = None) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            scoring: Scoring metrics

        Returns:
            Cross-validation scores
        """
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric)
            if 'neg_' in metric:
                # Convert negative scores to positive
                scores = -scores
                metric_name = metric.replace('neg_', '')
            else:
                metric_name = metric

            cv_results[f'{metric_name}_mean'] = scores.mean()
            cv_results[f'{metric_name}_std'] = scores.std()

        self.cv_scores = cv_results
        return cv_results

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.

        Returns:
            DataFrame with feature importance or None
        """
        if not self.is_fitted:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df

        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
            return importance_df

        return None

    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance()
        if importance_df is None:
            print(f"Feature importance not available for {self.name}")
            return

        plt.figure(figsize=figsize)
        top_features = importance_df.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {self.name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, set_name: str) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            set_name: Name of the dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        return metrics

    def plot_predictions(self, X: pd.DataFrame, y_true: pd.Series,
                        title: str = None, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot predictions vs actual values.

        Args:
            X: Features
            y_true: True values
            title: Plot title
            figsize: Figure size
        """
        if not self.is_fitted:
            print(f"Model {self.name} is not fitted yet.")
            return

        y_pred = self.predict(X)

        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title or f'Predictions vs Actual - {self.name}')

        # Add R² score to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_residuals(self, X: pd.DataFrame, y_true: pd.Series,
                      figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot residual analysis.

        Args:
            X: Features
            y_true: True values
            figsize: Figure size
        """
        if not self.is_fitted:
            print(f"Model {self.name} is not fitted yet.")
            return

        y_pred = self.predict(X)
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')

        # Residuals distribution
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')

        plt.suptitle(f'Residual Analysis - {self.name}')
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: Path) -> None:
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet.")

        model_data = {
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'cv_scores': self.cv_scores
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> 'BaseModel':
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)

        # Create instance
        instance = cls(model_data['name'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_metrics = model_data['training_metrics']
        instance.validation_metrics = model_data['validation_metrics']
        instance.cv_scores = model_data['cv_scores']
        instance.is_fitted = True

        return instance

    def get_summary(self) -> Dict[str, Any]:
        """
        Get model summary.

        Returns:
            Dictionary with model summary
        """
        summary = {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'cv_scores': self.cv_scores
        }

        return summary

    def __str__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} (status: {status})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
```

#### 4.1.2 Create Specific Model Implementations
Create `src/models/linear_models.py`:
```python
"""
Linear regression models for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from .base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear Regression model."""

    def __init__(self, **kwargs):
        super().__init__("Linear Regression", **kwargs)
        self.model = self._create_model(**kwargs)

    def _create_model(self, **kwargs) -> LinearRegression:
        return LinearRegression(**kwargs)


class RidgeRegressionModel(BaseModel):
    """Ridge Regression model."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Ridge Regression", **kwargs)
        self.alpha = alpha
        self.model = self._create_model(alpha=alpha, **kwargs)

    def _create_model(self, alpha: float = 1.0, **kwargs) -> Ridge:
        return Ridge(alpha=alpha, **kwargs)


class LassoRegressionModel(BaseModel):
    """Lasso Regression model."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Lasso Regression", **kwargs)
        self.alpha = alpha
        self.model = self._create_model(alpha=alpha, **kwargs)

    def _create_model(self, alpha: float = 1.0, **kwargs) -> Lasso:
        return Lasso(alpha=alpha, **kwargs)


class ElasticNetModel(BaseModel):
    """ElasticNet Regression model."""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs):
        super().__init__("ElasticNet Regression", **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = self._create_model(alpha=alpha, l1_ratio=l1_ratio, **kwargs)

    def _create_model(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs) -> ElasticNet:
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)


class PolynomialRegressionModel(BaseModel):
    """Polynomial Regression model."""

    def __init__(self, degree: int = 2, alpha: float = 1.0, **kwargs):
        super().__init__("Polynomial Regression", **kwargs)
        self.degree = degree
        self.alpha = alpha
        self.model = self._create_model(degree=degree, alpha=alpha, **kwargs)

    def _create_model(self, degree: int = 2, alpha: float = 1.0, **kwargs) -> Pipeline:
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge', Ridge(alpha=alpha))
        ])
```

Create `src/models/ensemble_models.py`:
```python
"""
Ensemble models for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest Regression model."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 random_state: int = 42, **kwargs):
        super().__init__("Random Forest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self._create_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> RandomForestRegressor:
        return RandomForestRegressor(**kwargs)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Regression model."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, random_state: int = 42, **kwargs):
        super().__init__("Gradient Boosting", **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self._create_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(**kwargs)


class XGBoostModel(BaseModel):
    """XGBoost Regression model."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, random_state: int = 42, **kwargs):
        super().__init__("XGBoost", **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self._create_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(**kwargs)


class ExtraTreesModel(BaseModel):
    """Extra Trees Regression model."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 random_state: int = 42, **kwargs):
        super().__init__("Extra Trees", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = self._create_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def _create_model(self, **kwargs) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(**kwargs)
```

**Test**: Base model functionality
```python
python -c "
from src.models.linear_models import LinearRegressionModel
from src.models.ensemble_models import RandomForestModel
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
y = pd.Series(X.sum(axis=1) + np.random.randn(100) * 0.1)

# Test linear model
linear_model = LinearRegressionModel()
linear_model.fit(X, y)
predictions = linear_model.predict(X)

print('✓ Linear model working correctly')
print(f'Model fitted: {linear_model.is_fitted}')
print(f'R² score: {linear_model.training_metrics[\"r2\"]:.3f}')

# Test ensemble model
rf_model = RandomForestModel(n_estimators=10)
rf_model.fit(X, y)
rf_predictions = rf_model.predict(X)

print('✓ Random Forest model working correctly')
print(f'Feature importance available: {rf_model.get_feature_importance() is not None}')
"
```

### 4.2 Model Training Pipeline

#### 4.2.1 Create Model Training Manager
Create `src/model_training.py`:
```python
"""
Model training and evaluation pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Type
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config.settings import MODELS_DIR, RANDOM_STATE
from src.models.base_model import BaseModel
from src.models.linear_models import (
    LinearRegressionModel, RidgeRegressionModel, LassoRegressionModel,
    ElasticNetModel, PolynomialRegressionModel
)
from src.models.ensemble_models import (
    RandomForestModel, GradientBoostingModel, XGBoostModel, ExtraTreesModel
)


class ModelTrainer:
    """Comprehensive model training and evaluation system."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series):
        """
        Initialize model trainer.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Model registry
        self.models = {}
        self.trained_models = {}
        self.model_results = {}

        # Default hyperparameter grids
        self.hyperparameter_grids = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'elasticnet': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        }

    def register_model(self, model_class: Type[BaseModel], name: str, **kwargs) -> None:
        """
        Register a model for training.

        Args:
            model_class: Model class to instantiate
            name: Model name
            **kwargs: Model parameters
        """
        self.models[name] = (model_class, kwargs)

    def register_default_models(self) -> None:
        """Register default set of models."""
        self.register_model(LinearRegressionModel, 'linear')
        self.register_model(RidgeRegressionModel, 'ridge', alpha=1.0)
        self.register_model(LassoRegressionModel, 'lasso', alpha=0.1)
        self.register_model(ElasticNetModel, 'elasticnet', alpha=0.1, l1_ratio=0.5)
        self.register_model(RandomForestModel, 'random_forest',
                          n_estimators=100, max_depth=20, random_state=RANDOM_STATE)
        self.register_model(GradientBoostingModel, 'gradient_boosting',
                          n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)
        self.register_model(XGBoostModel, 'xgboost',
                          n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE)

    def train_single_model(self, model_name: str, tune_hyperparameters: bool = False,
                          cv_folds: int = 5) -> BaseModel:
        """
        Train a single model.

        Args:
            model_name: Name of the model to train
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds

        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")

        model_class, base_kwargs = self.models[model_name]

        if tune_hyperparameters and model_name in self.hyperparameter_grids:
            # Hyperparameter tuning
            print(f"Tuning hyperparameters for {model_name}...")
            best_model = self._tune_hyperparameters(model_class, model_name, base_kwargs, cv_folds)
        else:
            # Train with default parameters
            print(f"Training {model_name} with default parameters...")
            best_model = model_class(**base_kwargs)
            best_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        # Perform cross-validation
        cv_scores = best_model.cross_validate(self.X_train, self.y_train, cv=cv_folds)

        # Store results
        self.trained_models[model_name] = best_model
        self.model_results[model_name] = {
            'model': best_model,
            'training_metrics': best_model.training_metrics,
            'validation_metrics': best_model.validation_metrics,
            'cv_scores': cv_scores
        }

        print(f"✓ {model_name} training completed")
        return best_model

    def train_all_models(self, tune_hyperparameters: bool = False,
                        cv_folds: int = 5) -> Dict[str, BaseModel]:
        """
        Train all registered models.

        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of trained models
        """
        if not self.models:
            self.register_default_models()

        print(f"Training {len(self.models)} models...")
        print("="*50)

        for model_name in self.models.keys():
            try:
                self.train_single_model(model_name, tune_hyperparameters, cv_folds)
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")

        print("\n✅ All models training completed")
        return self.trained_models

    def _tune_hyperparameters(self, model_class: Type[BaseModel], model_name: str,
                             base_kwargs: Dict, cv_folds: int) -> BaseModel:
        """
        Tune hyperparameters using GridSearchCV.

        Args:
            model_class: Model class
            model_name: Model name
            base_kwargs: Base model parameters
            cv_folds: Cross-validation folds

        Returns:
            Best model
        """
        # Create base model for hyperparameter tuning
        base_model = model_class(**base_kwargs)
        param_grid = self.hyperparameter_grids[model_name]

        # Use the underlying sklearn model for GridSearch
        grid_search = GridSearchCV(
            base_model.model,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)

        # Create best model with optimal parameters
        best_params = {**base_kwargs, **grid_search.best_params_}
        best_model = model_class(**best_params)
        best_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        print(f"  Best parameters for {model_name}: {grid_search.best_params_}")
        return best_model

    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all trained models and create comparison.

        Returns:
            DataFrame with model comparison
        """
        if not self.trained_models:
            raise ValueError("No models have been trained yet")

        evaluation_data = []

        for model_name, model in self.trained_models.items():
            # Get predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))

            train_r2 = r2_score(self.y_train, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)

            # Get CV scores
            cv_rmse = model.cv_scores.get('mean_squared_error_mean', np.nan)
            cv_rmse = np.sqrt(cv_rmse) if not np.isnan(cv_rmse) else np.nan

            evaluation_data.append({
                'model': model_name,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'cv_rmse': cv_rmse,
                'overfit_ratio': val_rmse / train_rmse if train_rmse > 0 else np.inf
            })

        evaluation_df = pd.DataFrame(evaluation_data)
        evaluation_df = evaluation_df.sort_values('val_rmse')

        return evaluation_df

    def get_best_model(self, metric: str = 'val_rmse') -> Tuple[str, BaseModel]:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for selection

        Returns:
            Tuple of (model_name, model)
        """
        evaluation_df = self.evaluate_models()

        if metric in ['val_rmse', 'train_rmse', 'cv_rmse', 'overfit_ratio']:
            best_model_name = evaluation_df.loc[evaluation_df[metric].idxmin(), 'model']
        else:  # For R² scores (higher is better)
            best_model_name = evaluation_df.loc[evaluation_df[metric].idxmax(), 'model']

        return best_model_name, self.trained_models[best_model_name]

    def plot_model_comparison(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot model comparison visualizations.

        Args:
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        evaluation_df = self.evaluate_models()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # RMSE comparison
        axes[0,0].bar(evaluation_df['model'], evaluation_df['val_rmse'])
        axes[0,0].set_title('Validation RMSE Comparison')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].tick_params(axis='x', rotation=45)

        # R² comparison
        axes[0,1].bar(evaluation_df['model'], evaluation_df['val_r2'])
        axes[0,1].set_title('Validation R² Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Overfitting analysis
        axes[1,0].scatter(evaluation_df['train_rmse'], evaluation_df['val_rmse'])
        for i, model in enumerate(evaluation_df['model']):
            axes[1,0].annotate(model, (evaluation_df.iloc[i]['train_rmse'], evaluation_df.iloc[i]['val_rmse']))
        axes[1,0].plot([evaluation_df['train_rmse'].min(), evaluation_df['train_rmse'].max()],
                      [evaluation_df['train_rmse'].min(), evaluation_df['train_rmse'].max()], 'r--')
        axes[1,0].set_xlabel('Training RMSE')
        axes[1,0].set_ylabel('Validation RMSE')
        axes[1,0].set_title('Overfitting Analysis')

        # Cross-validation scores
        axes[1,1].bar(evaluation_df['model'], evaluation_df['cv_rmse'])
        axes[1,1].set_title('Cross-Validation RMSE')
        axes[1,1].set_ylabel('CV RMSE')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def save_models(self, models_dir: Path = None) -> None:
        """
        Save all trained models.

        Args:
            models_dir: Directory to save models
        """
        if models_dir is None:
            models_dir = MODELS_DIR

        models_dir.mkdir(exist_ok=True, parents=True)

        # Save individual models
        for model_name, model in self.trained_models.items():
            model_path = models_dir / f"{model_name}_model.pkl"
            model.save_model(model_path)

        # Save evaluation results
        if self.trained_models:
            evaluation_df = self.evaluate_models()
            evaluation_df.to_csv(models_dir / "model_evaluation.csv", index=False)

        # Save training summary
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.trained_models.keys()),
            'training_data_shape': self.X_train.shape,
            'validation_data_shape': self.X_val.shape,
            'best_model': self.get_best_model()[0]
        }

        joblib.dump(training_summary, models_dir / "training_summary.pkl")
        print(f"Models saved to: {models_dir}")

    def generate_training_report(self) -> str:
        """
        Generate comprehensive training report.

        Returns:
            Training report string
        """
        if not self.trained_models:
            return "No models have been trained yet."

        evaluation_df = self.evaluate_models()
        best_model_name, best_model = self.get_best_model()

        report = []
        report.append("=" * 60)
        report.append("MODEL TRAINING REPORT")
        report.append("=" * 60)

        # Training overview
        report.append(f"\nTRAINING OVERVIEW:")
        report.append(f"Models trained: {len(self.trained_models)}")
        report.append(f"Training samples: {self.X_train.shape[0]}")
        report.append(f"Validation samples: {self.X_val.shape[0]}")
        report.append(f"Features: {self.X_train.shape[1]}")

        # Best model
        report.append(f"\nBEST MODEL: {best_model_name}")
        best_metrics = evaluation_df[evaluation_df['model'] == best_model_name].iloc[0]
        report.append(f"Validation RMSE: {best_metrics['val_rmse']:.4f}")
        report.append(f"Validation R²: {best_metrics['val_r2']:.4f}")
        report.append(f"Cross-validation RMSE: {best_metrics['cv_rmse']:.4f}")

        # All models performance
        report.append(f"\nALL MODELS PERFORMANCE:")
        report.append(evaluation_df.to_string(index=False, float_format='%.4f'))

        # Feature importance (if available)
        importance_df = best_model.get_feature_importance()
        if importance_df is not None:
            report.append(f"\nTOP 10 FEATURES ({best_model_name}):")
            for _, row in importance_df.head(10).iterrows():
                report.append(f"  {row['feature']}: {row['importance']:.4f}")

        return "\n".join(report)
```

**Test**: Model training pipeline
```python
python -c "
from src.model_training import ModelTrainer
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 1000
n_features = 10

X = pd.DataFrame(np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)])
y = pd.Series(X.sum(axis=1) + np.random.randn(n_samples) * 0.5)

# Split data
split_idx = int(0.8 * n_samples)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Test model trainer
trainer = ModelTrainer(X_train, y_train, X_val, y_val)
trainer.register_default_models()

# Train a few models
trained_models = trainer.train_all_models(tune_hyperparameters=False)
evaluation_df = trainer.evaluate_models()
best_model_name, best_model = trainer.get_best_model()

print('✓ Model training pipeline working correctly')
print(f'Models trained: {len(trained_models)}')
print(f'Best model: {best_model_name}')
print(f'Best validation RMSE: {evaluation_df.iloc[0][\"val_rmse\"]:.4f}')
"
```

### 4.3 Model Validation and Selection

#### 4.3.1 Create Model Validation Module
Create `src/model_validation.py`:
```python
"""
Model validation and selection utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.base_model import BaseModel


class ModelValidator:
    """Comprehensive model validation and analysis."""

    def __init__(self, models: Dict[str, BaseModel], X_train: pd.DataFrame,
                 y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Initialize model validator.

        Args:
            models: Dictionary of trained models
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def validate_single_model(self, model: BaseModel, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive validation for a single model.

        Args:
            model: Model to validate
            cv_folds: Number of cross-validation folds

        Returns:
            Validation results dictionary
        """
        results = {}

        # Basic predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)

        # Basic metrics
        results['train_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
            'mae': mean_absolute_error(self.y_train, train_pred),
            'r2': r2_score(self.y_train, train_pred),
            'mape': np.mean(np.abs((self.y_train - train_pred) / self.y_train)) * 100
        }

        results['val_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(self.y_val, val_pred)),
            'mae': mean_absolute_error(self.y_val, val_pred),
            'r2': r2_score(self.y_val, val_pred),
            'mape': np.mean(np.abs((self.y_val - val_pred) / self.y_val)) * 100
        }

        # Cross-validation
        cv_scores = cross_val_score(model.model, self.X_train, self.y_train,
                                   cv=cv_folds, scoring='neg_mean_squared_error')
        results['cv_metrics'] = {
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'cv_scores': cv_scores
        }

        # Overfitting analysis
        results['overfitting'] = {
            'rmse_ratio': results['val_metrics']['rmse'] / results['train_metrics']['rmse'],
            'r2_difference': results['train_metrics']['r2'] - results['val_metrics']['r2'],
            'is_overfitting': results['val_metrics']['rmse'] / results['train_metrics']['rmse'] > 1.1
        }

        # Residual analysis
        train_residuals = self.y_train - train_pred
        val_residuals = self.y_val - val_pred

        results['residual_analysis'] = {
            'train_residuals': train_residuals,
            'val_residuals': val_residuals,
            'train_residual_std': train_residuals.std(),
            'val_residual_std': val_residuals.std(),
            'residual_normality': self._test_normality(val_residuals)
        }

        return results

    def compare_models(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare all models across multiple metrics.

        Args:
            metrics: Metrics to include in comparison

        Returns:
            Comparison DataFrame
        """
        if metrics is None:
            metrics = ['val_rmse', 'val_r2', 'cv_rmse_mean', 'rmse_ratio']

        comparison_data = []

        for model_name, model in self.models.items():
            validation_results = self.validate_single_model(model)

            row = {'model': model_name}

            # Extract requested metrics
            for metric in metrics:
                if metric.startswith('val_'):
                    row[metric] = validation_results['val_metrics'][metric.replace('val_', '')]
                elif metric.startswith('train_'):
                    row[metric] = validation_results['train_metrics'][metric.replace('train_', '')]
                elif metric.startswith('cv_'):
                    row[metric] = validation_results['cv_metrics'][metric]
                elif metric in validation_results['overfitting']:
                    row[metric] = validation_results['overfitting'][metric]

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('val_rmse')

    def plot_model_performance(self, figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Create comprehensive performance plots.

        Args:
            figsize: Figure size
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        # Get validation results for all models
        all_results = {}
        for model_name, model in self.models.items():
            all_results[model_name] = self.validate_single_model(model)

        # 1. Validation RMSE comparison
        models = list(self.models.keys())
        val_rmse = [all_results[model]['val_metrics']['rmse'] for model in models]

        axes[0].bar(models, val_rmse)
        axes[0].set_title('Validation RMSE Comparison')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)

        # 2. Validation R² comparison
        val_r2 = [all_results[model]['val_metrics']['r2'] for model in models]

        axes[1].bar(models, val_r2)
        axes[1].set_title('Validation R² Comparison')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)

        # 3. Overfitting analysis
        train_rmse = [all_results[model]['train_metrics']['rmse'] for model in models]
        val_rmse = [all_results[model]['val_metrics']['rmse'] for model in models]

        axes[2].scatter(train_rmse, val_rmse)
        for i, model in enumerate(models):
            axes[2].annotate(model, (train_rmse[i], val_rmse[i]))

        min_rmse = min(min(train_rmse), min(val_rmse))
        max_rmse = max(max(train_rmse), max(val_rmse))
        axes[2].plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'r--', alpha=0.8)
        axes[2].set_xlabel('Training RMSE')
        axes[2].set_ylabel('Validation RMSE')
        axes[2].set_title('Overfitting Analysis')

        # 4. Cross-validation scores
        cv_means = [all_results[model]['cv_metrics']['cv_rmse_mean'] for model in models]
        cv_stds = [all_results[model]['cv_metrics']['cv_rmse_std'] for model in models]

        axes[3].bar(models, cv_means, yerr=cv_stds, capsize=5)
        axes[3].set_title('Cross-Validation RMSE')
        axes[3].set_ylabel('CV RMSE')
        axes[3].tick_params(axis='x', rotation=45)

        # 5-8. Individual model predictions vs actual (show first 4 models)
        for i, (model_name, model) in enumerate(list(self.models.items())[:4]):
            ax_idx = 4 + i
            val_pred = model.predict(self.X_val)

            axes[ax_idx].scatter(self.y_val, val_pred, alpha=0.6)
            axes[ax_idx].plot([self.y_val.min(), self.y_val.max()],
                             [self.y_val.min(), self.y_val.max()], 'r--', lw=2)
            axes[ax_idx].set_xlabel('Actual')
            axes[ax_idx].set_ylabel('Predicted')
            axes[ax_idx].set_title(f'{model_name} - Predictions vs Actual')

            # Add R² to plot
            r2 = r2_score(self.y_val, val_pred)
            axes[ax_idx].text(0.05, 0.95, f'R² = {r2:.3f}',
                             transform=axes[ax_idx].transAxes,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Hide unused subplots
        for i in range(len(self.models) + 4, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_residual_analysis(self, model_name: str, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot detailed residual analysis for a specific model.

        Args:
            model_name: Name of the model to analyze
            figsize: Figure size
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        validation_results = self.validate_single_model(model)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        val_pred = model.predict(self.X_val)
        residuals = validation_results['residual_analysis']['val_residuals']

        # Residuals vs Predicted
        axes[0].scatter(val_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')

        # Residuals distribution
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')

        plt.suptitle(f'Residual Analysis - {model_name}')
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, model_name: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot learning curves for a specific model.

        Args:
            model_name: Name of the model to analyze
            figsize: Figure size
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        train_sizes, train_scores, val_scores = learning_curve(
            model.model, self.X_train, self.y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )

        # Convert to RMSE and make positive
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)

        train_rmse_mean = train_rmse.mean(axis=1)
        train_rmse_std = train_rmse.std(axis=1)
        val_rmse_mean = val_rmse.mean(axis=1)
        val_rmse_std = val_rmse.std(axis=1)

        plt.figure(figsize=figsize)
        plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                        train_rmse_mean + train_rmse_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                        val_rmse_mean + val_rmse_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_rmse_mean, 'o-', color="r", label="Training RMSE")
        plt.plot(train_sizes, val_rmse_mean, 'o-', color="g", label="Validation RMSE")

        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def _test_normality(self, residuals: pd.Series) -> Dict[str, float]:
        """
        Test residuals for normality.

        Args:
            residuals: Residuals to test

        Returns:
            Normality test results
        """
        from scipy import stats

        # Shapiro-Wilk test (for smaller samples)
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            return {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal_shapiro': shapiro_p > 0.05
            }
        else:
            # Use Kolmogorov-Smirnov test for larger samples
            ks_stat, ks_p = stats.kstest(residuals, 'norm')
            return {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'is_normal_ks': ks_p > 0.05
            }

    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report.

        Returns:
            Validation report string
        """
        comparison_df = self.compare_models()

        report = []
        report.append("=" * 60)
        report.append("MODEL VALIDATION REPORT")
        report.append("=" * 60)

        # Model ranking
        report.append(f"\nMODEL RANKING (by Validation RMSE):")
        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            report.append(f"{i}. {row['model']}: {row['val_rmse']:.4f}")

        # Best model details
        best_model_name = comparison_df.iloc[0]['model']
        best_model = self.models[best_model_name]
        best_results = self.validate_single_model(best_model)

        report.append(f"\nBEST MODEL ANALYSIS: {best_model_name}")
        report.append(f"Validation RMSE: {best_results['val_metrics']['rmse']:.4f}")
        report.append(f"Validation R²: {best_results['val_metrics']['r2']:.4f}")
        report.append(f"Validation MAE: {best_results['val_metrics']['mae']:.4f}")
        report.append(f"Cross-validation RMSE: {best_results['cv_metrics']['cv_rmse_mean']:.4f} ± {best_results['cv_metrics']['cv_rmse_std']:.4f}")

        # Overfitting analysis
        report.append(f"\nOVERFITTING ANALYSIS:")
        report.append(f"RMSE Ratio (Val/Train): {best_results['overfitting']['rmse_ratio']:.3f}")
        report.append(f"R² Difference (Train-Val): {best_results['overfitting']['r2_difference']:.3f}")
        report.append(f"Overfitting detected: {'Yes' if best_results['overfitting']['is_overfitting'] else 'No'}")

        # Model comparison table
        report.append(f"\nFULL MODEL COMPARISON:")
        report.append(comparison_df.to_string(index=False, float_format='%.4f'))

        return "\n".join(report)
```

### 4.4 Model Training Notebook

#### 4.4.1 Create Training Notebook
Create `notebooks/03_model_development_training.ipynb`:

```python
# Cell 1: Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from src.model_training import ModelTrainer
from src.model_validation import ModelValidator
from src.data_pipeline import DataPreprocessingPipeline

# Import models
from src.models.linear_models import *
from src.models.ensemble_models import *

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("Model Development Environment Setup Complete")
```

```python
# Cell 2: Load Processed Data
try:
    # Try to load previously processed data
    pipeline = DataPreprocessingPipeline(pd.DataFrame(), pd.DataFrame())
    datasets = pipeline.load_processed_data()

    if 'X_train' in datasets and 'X_val' in datasets:
        X_train = datasets['X_train']
        X_val = datasets['X_val']
        y_train = datasets['y_train']
        y_val = datasets['y_val']
        print("✓ Processed data loaded successfully")
    else:
        raise FileNotFoundError("Processed data not found")

except FileNotFoundError:
    print("Processed data not found. Creating sample data for demonstration...")
    # Create comprehensive sample data for model training
    np.random.seed(42)
    n_samples = 2000
    n_features = 50

    # Generate features with different characteristics
    X_data = np.random.randn(n_samples, n_features)

    # Create some correlated features
    X_data[:, 1] = X_data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
    X_data[:, 2] = X_data[:, 0] * 0.6 + X_data[:, 1] * 0.4 + np.random.randn(n_samples) * 0.3

    # Create target with non-linear relationships
    y_data = (2 * X_data[:, 0] +
              1.5 * X_data[:, 1] +
              0.8 * X_data[:, 2] +
              0.5 * X_data[:, 3] * X_data[:, 4] +  # Interaction
              0.3 * X_data[:, 5] ** 2 +  # Non-linear
              np.random.randn(n_samples) * 0.5)

    # Convert to DataFrames
    X_all = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(n_features)])
    y_all = pd.Series(y_data, name='target')

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    print(f"Sample data created - Training: {X_train.shape}, Validation: {X_val.shape}")

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Feature names (first 10): {list(X_train.columns[:10])}")
```

```python
# Cell 3: Initialize Model Trainer and Register Models
print("Initializing model trainer...")
trainer = ModelTrainer(X_train, y_train, X_val, y_val)

# Register all models
print("Registering models...")
trainer.register_default_models()

# Also register some additional models with different parameters
trainer.register_model(RidgeRegressionModel, 'ridge_strong', alpha=10.0)
trainer.register_model(LassoRegressionModel, 'lasso_strong', alpha=1.0)
trainer.register_model(ElasticNetModel, 'elasticnet_balanced', alpha=0.5, l1_ratio=0.5)
trainer.register_model(ExtraTreesModel, 'extra_trees', n_estimators=200, max_depth=15, random_state=42)

print(f"Total models registered: {len(trainer.models)}")
print("Registered models:", list(trainer.models.keys()))
```

```python
# Cell 4: Train All Models
print("Starting model training...")
print("=" * 60)

# Train without hyperparameter tuning first (faster)
trained_models = trainer.train_all_models(tune_hyperparameters=False, cv_folds=5)

print(f"\n✅ Training completed for {len(trained_models)} models")
```

```python
# Cell 5: Evaluate and Compare Models
print("Evaluating model performance...")

# Get evaluation results
evaluation_df = trainer.evaluate_models()
print("\nModel Performance Comparison:")
print(evaluation_df.round(4))

# Get best model
best_model_name, best_model = trainer.get_best_model()
print(f"\n🏆 Best Model: {best_model_name}")
print(f"Best Validation RMSE: {evaluation_df.iloc[0]['val_rmse']:.4f}")
print(f"Best Validation R²: {evaluation_df.iloc[0]['val_r2']:.4f}")
```

```python
# Cell 6: Visualize Model Comparison
print("Creating model comparison visualizations...")
trainer.plot_model_comparison(figsize=(16, 12))
```

```python
# Cell 7: Detailed Model Validation
print("Performing detailed model validation...")

# Initialize validator
validator = ModelValidator(trained_models, X_train, y_train, X_val, y_val)

# Create comprehensive performance plots
validator.plot_model_performance(figsize=(18, 15))
```

```python
# Cell 8: Best Model Analysis
print(f"Analyzing best model: {best_model_name}")

# Plot feature importance
print("Feature Importance:")
best_model.plot_feature_importance(top_n=20, figsize=(12, 10))

# Plot predictions vs actual
best_model.plot_predictions(X_val, y_val, title=f'{best_model_name} - Validation Set')

# Plot residual analysis
validator.plot_residual_analysis(best_model_name, figsize=(15, 5))
```

```python
# Cell 9: Learning Curves Analysis
print("Analyzing learning curves for top 3 models...")

top_3_models = evaluation_df.head(3)['model'].tolist()

for model_name in top_3_models:
    print(f"\nLearning curves for: {model_name}")
    validator.plot_learning_curves(model_name, figsize=(10, 6))
```

```python
# Cell 10: Hyperparameter Tuning for Best Models
print("Performing hyperparameter tuning for top 3 models...")

# Create new trainer for tuning
tuning_trainer = ModelTrainer(X_train, y_train, X_val, y_val)

# Register only top 3 models for tuning
for model_name in top_3_models:
    if model_name in ['ridge', 'lasso', 'elasticnet', 'random_forest', 'gradient_boosting', 'xgboost']:
        model_class, kwargs = trainer.models[model_name]
        tuning_trainer.register_model(model_class, f'{model_name}_tuned', **kwargs)

# Train with hyperparameter tuning
print("Training with hyperparameter tuning...")
tuned_models = tuning_trainer.train_all_models(tune_hyperparameters=True, cv_folds=5)

# Compare tuned vs original
if tuned_models:
    tuned_evaluation = tuning_trainer.evaluate_models()
    print("\nTuned Models Performance:")
    print(tuned_evaluation.round(4))
```

```python
# Cell 11: Final Model Selection and Evaluation
print("Final model selection...")

# Combine all models for final comparison
all_models = {**trained_models, **tuned_models}

# Final validation
final_validator = ModelValidator(all_models, X_train, y_train, X_val, y_val)
final_comparison = final_validator.compare_models()

print("FINAL MODEL COMPARISON:")
print(final_comparison.round(4))

# Select final best model
final_best_name = final_comparison.iloc[0]['model']
final_best_model = all_models[final_best_name]

print(f"\n🎯 FINAL BEST MODEL: {final_best_name}")
print(f"Final Validation RMSE: {final_comparison.iloc[0]['val_rmse']:.4f}")
print(f"Final Validation R²: {final_comparison.iloc[0]['val_r2']:.4f}")
```

```python
# Cell 12: Generate and Display Reports
print("Generating comprehensive reports...")

# Training report
training_report = trainer.generate_training_report()
print(training_report)

print("\n" + "="*80 + "\n")

# Validation report
validation_report = final_validator.generate_validation_report()
print(validation_report)
```

```python
# Cell 13: Save Models and Results
print("Saving models and results...")

# Save all trained models
trainer.save_models()

# Save tuned models if any
if tuned_models:
    tuning_trainer.save_models()

# Save final comparison results
final_comparison.to_csv('../models/final_model_comparison.csv', index=False)

print("✅ All models and results saved successfully!")
print("Models are ready for deployment in the web application.")
```

### 4.5 Testing and Validation

Create `tests/test_phase4.py`:
```python
"""
Tests for Phase 4: Model Development & Training
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append('..')

from src.models.linear_models import LinearRegressionModel, RidgeRegressionModel
from src.models.ensemble_models import RandomForestModel, XGBoostModel
from src.model_training import ModelTrainer
from src.model_validation import ModelValidator


class TestPhase4(unittest.TestCase):
    """Test Phase 4 functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create synthetic data
        self.X = pd.DataFrame(np.random.randn(n_samples, n_features),
                             columns=[f'feature_{i}' for i in range(n_features)])
        self.y = pd.Series(self.X.sum(axis=1) + np.random.randn(n_samples) * 0.1)

        # Split data
        split_idx = int(0.8 * n_samples)
        self.X_train, self.X_val = self.X[:split_idx], self.X[split_idx:]
        self.y_train, self.y_val = self.y[:split_idx], self.y[split_idx:]

    def test_base_model_functionality(self):
        """Test base model functionality."""
        # Test linear regression model
        model = LinearRegressionModel()
        self.assertEqual(model.name, "Linear Regression")
        self.assertFalse(model.is_fitted)

        # Test fitting
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.feature_names)
        self.assertEqual(len(model.feature_names), self.X_train.shape[1])

        # Test predictions
        predictions = model.predict(self.X_val)
        self.assertEqual(len(predictions), len(self.y_val))

        # Test metrics calculation
        self.assertIn('rmse', model.training_metrics)
        self.assertIn('r2', model.training_metrics)
        self.assertIn('rmse', model.validation_metrics)

    def test_ensemble_models(self):
        """Test ensemble models."""
        # Test Random Forest
        rf_model = RandomForestModel(n_estimators=10, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_val)
        self.assertEqual(len(rf_pred), len(self.y_val))

        # Test feature importance
        importance_df = rf_model.get_feature_importance()
        self.assertIsNotNone(importance_df)
        self.assertEqual(len(importance_df), self.X_train.shape[1])

        # Test XGBoost
        xgb_model = XGBoostModel(n_estimators=10, random_state=42)
        xgb_model.fit(self.X_train, self.y_train)
        xgb_pred = xgb_model.predict(self.X_val)
        self.assertEqual(len(xgb_pred), len(self.y_val))

    def test_model_trainer(self):
        """Test model training pipeline."""
        trainer = ModelTrainer(self.X_train, self.y_train, self.X_val, self.y_val)

        # Test model registration
        trainer.register_model(LinearRegressionModel, 'linear_test')
        trainer.register_model(RidgeRegressionModel, 'ridge_test', alpha=1.0)
        self.assertEqual(len(trainer.models), 2)

        # Test single model training
        model = trainer.train_single_model('linear_test')
        self.assertIsInstance(model, LinearRegressionModel)
        self.assertTrue(model.is_fitted)

        # Test evaluation
        evaluation_df = trainer.evaluate_models()
        self.assertIsInstance(evaluation_df, pd.DataFrame)
        self.assertIn('model', evaluation_df.columns)
        self.assertIn('val_rmse', evaluation_df.columns)

        # Test best model selection
        best_name, best_model = trainer.get_best_model()
        self.assertIn(best_name, trainer.trained_models)
        self.assertIsInstance(best_model, LinearRegressionModel)

    def test_model_validation(self):
        """Test model validation."""
        # Train a few models first
        models = {
            'linear': LinearRegressionModel(),
            'ridge': RidgeRegressionModel(alpha=1.0)
        }

        for model in models.values():
            model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        # Test validator
        validator = ModelValidator(models, self.X_train, self.y_train, self.X_val, self.y_val)

        # Test single model validation
        validation_results = validator.validate_single_model(models['linear'])
        self.assertIn('train_metrics', validation_results)
        self.assertIn('val_metrics', validation_results)
        self.assertIn('cv_metrics', validation_results)
        self.assertIn('overfitting', validation_results)

        # Test model comparison
        comparison_df = validator.compare_models()
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), len(models))

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        model = LinearRegressionModel()
        model.fit(self.X_train, self.y_train)

        # Test cross-validation
        cv_scores = model.cross_validate(self.X_train, self.y_train, cv=3)
        self.assertIsInstance(cv_scores, dict)
        self.assertIn('mean_squared_error_mean', cv_scores)

    def test_model_serialization(self):
        """Test model saving and loading."""
        # Train a model
        model = RidgeRegressionModel(alpha=1.0)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        # Test saving
        model_path = Path('test_model.pkl')
        try:
            model.save_model(model_path)
            self.assertTrue(model_path.exists())

            # Test loading
            loaded_model = RidgeRegressionModel.load_model(model_path)
            self.assertTrue(loaded_model.is_fitted)
            self.assertEqual(loaded_model.name, model.name)

            # Test predictions are the same
            original_pred = model.predict(self.X_val)
            loaded_pred = loaded_model.predict(self.X_val)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        finally:
            # Clean up
            if model_path.exists():
                model_path.unlink()


if __name__ == '__main__':
    unittest.main()
```

**Test**: Run Phase 4 tests
```bash
cd tests
python test_phase4.py
```

## Deliverables
- [ ] Base model class with common functionality
- [ ] Linear regression model implementations
- [ ] Ensemble model implementations (Random Forest, XGBoost, etc.)
- [ ] Comprehensive model training pipeline
- [ ] Model validation and comparison system
- [ ] Hyperparameter tuning capabilities
- [ ] Cross-validation implementation
- [ ] Model performance visualization
- [ ] Model serialization and loading
- [ ] Comprehensive model training notebook
- [ ] Model evaluation reports
- [ ] Automated testing suite

## Success Criteria
- Multiple models trained successfully
- Model comparison shows clear performance differences
- Best model identified through proper validation
- Hyperparameter tuning improves model performance
- Cross-validation scores are consistent
- Models can be saved and loaded correctly
- All tests pass successfully
- Training pipeline runs end-to-end
- Performance visualizations provide clear insights

## Next Phase
Proceed to **Phase 5: Web Application & Deployment** with the trained and validated models ready for integration into the Streamlit application.