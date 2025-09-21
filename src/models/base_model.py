"""
Base model class for California housing price prediction.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import joblib
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class BaseModel(ABC):
    """Abstract base class for all California housing prediction models."""

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
        self.model_params = kwargs

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
        Train the model on California housing data.

        Args:
            X_train: Training features
            y_train: Training target (house values)
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Self for method chaining
        """
        print(f"🔧 Training {self.name}...")

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

        print(f"✅ {self.name} training completed")
        print(f"  • Training R²: {self.training_metrics['r2']:.3f}")
        if self.validation_metrics:
            print(f"  • Validation R²: {self.validation_metrics['r2']:.3f}")
            print(f"  • Validation RMSE: ${self.validation_metrics['rmse']:,.0f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make housing price predictions.

        Args:
            X: Features for prediction

        Returns:
            Predicted house values
        """
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet. Call fit() first.")

        return self.model.predict(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
                      scoring: List[str] = None) -> Dict[str, float]:
        """
        Perform cross-validation for California housing prediction.

        Args:
            X: Features
            y: Target house values
            cv: Number of cross-validation folds
            scoring: Scoring metrics

        Returns:
            Cross-validation scores
        """
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        print(f"🔄 Running {cv}-fold cross-validation for {self.name}...")

        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            if 'neg_' in metric:
                # Convert negative scores to positive
                scores = -scores
                metric_name = metric.replace('neg_', '')
            else:
                metric_name = metric

            cv_results[f'{metric_name}_mean'] = float(scores.mean())
            cv_results[f'{metric_name}_std'] = float(scores.std())

        self.cv_scores = cv_results

        print(f"  • CV RMSE: ${np.sqrt(cv_results['mean_squared_error_mean']):,.0f} ± ${np.sqrt(cv_results['mean_squared_error_std']):,.0f}")
        print(f"  • CV R²: {cv_results.get('r2_mean', 0):.3f} ± {cv_results.get('r2_std', 0):.3f}")

        return cv_results

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance for California housing features.

        Returns:
            DataFrame with feature importance or None
        """
        if not self.is_fitted:
            return None

        try:
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

                return importance_df

            elif hasattr(self.model, 'coef_'):
                # Linear models
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': np.abs(self.model.coef_)
                }).sort_values('importance', ascending=False)

                return importance_df

        except Exception as e:
            print(f"⚠️ Could not extract feature importance: {e}")

        return None

    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance for California housing prediction.

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

        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)

        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance - {self.name}')
        plt.gca().invert_yaxis()

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.01 * max(top_features['importance']),
                    bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}',
                    ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, X: pd.DataFrame, y_true: pd.Series,
                        title: str = None, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot predictions vs actual house values.

        Args:
            X: Features
            y_true: True house values
            title: Plot title
            figsize: Figure size
        """
        if not self.is_fitted:
            print(f"Model {self.name} is not fitted yet.")
            return

        y_pred = self.predict(X)

        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual House Values ($)')
        plt.ylabel('Predicted House Values ($)')
        plt.title(title or f'Predictions vs Actual - {self.name}')

        # Add R² score and RMSE to plot
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        plt.text(0.05, 0.95, f'R² = {r2:.3f}\\nRMSE = ${rmse:,.0f}',
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')

        # Format axes to show dollar amounts
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        plt.tight_layout()
        plt.show()

    def plot_residuals(self, X: pd.DataFrame, y_true: pd.Series,
                      figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot residual analysis for housing price predictions.

        Args:
            X: Features
            y_true: True house values
            figsize: Figure size
        """
        if not self.is_fitted:
            print(f"Model {self.name} is not fitted yet.")
            return

        y_pred = self.predict(X)
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values ($)')
        axes[0].set_ylabel('Residuals ($)')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Residuals distribution
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='lightblue')
        axes[1].set_xlabel('Residuals ($)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].axvline(residuals.mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: ${residuals.mean():,.0f}')
        axes[1].legend()

        # Q-Q plot for residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot (Residuals vs Normal)')

        plt.suptitle(f'Residual Analysis - {self.name}')
        plt.tight_layout()
        plt.show()

        # Print residual statistics
        print(f"📊 Residual Statistics for {self.name}:")
        print(f"  • Mean residual: ${residuals.mean():,.0f}")
        print(f"  • Std residual: ${residuals.std():,.0f}")
        print(f"  • Mean absolute residual: ${np.abs(residuals).mean():,.0f}")

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, set_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics for housing prediction.

        Args:
            y_true: True house values
            y_pred: Predicted house values
            set_name: Name of the dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        }

        # Additional housing-specific metrics
        price_range = y_true.max() - y_true.min()
        metrics['rmse_normalized'] = float(metrics['rmse'] / price_range * 100)  # RMSE as % of price range
        metrics['mean_price'] = float(y_true.mean())
        metrics['rmse_relative'] = float(metrics['rmse'] / metrics['mean_price'] * 100)  # RMSE as % of mean price

        return metrics

    def save_model(self, filepath: Path) -> None:
        """
        Save the trained housing price prediction model.

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
            'cv_scores': self.cv_scores,
            'model_params': self.model_params,
            'model_type': 'california_housing_predictor'
        }

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> 'BaseModel':
        """
        Load a saved California housing prediction model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)

        # Create instance with original parameters
        instance = cls(model_data['name'], **model_data.get('model_params', {}))
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_metrics = model_data['training_metrics']
        instance.validation_metrics = model_data['validation_metrics']
        instance.cv_scores = model_data['cv_scores']
        instance.is_fitted = True

        return instance

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.

        Returns:
            Dictionary with model summary
        """
        summary = {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'cv_scores': self.cv_scores,
            'model_params': self.model_params
        }

        return summary

    def print_performance_summary(self) -> None:
        """Print a concise performance summary."""
        print(f"\n{'='*50}")
        print(f"PERFORMANCE SUMMARY - {self.name}")
        print(f"{'='*50}")

        if self.training_metrics:
            print(f"Training Performance:")
            print(f"  • R²: {self.training_metrics['r2']:.3f}")
            print(f"  • RMSE: ${self.training_metrics['rmse']:,.0f}")
            print(f"  • MAE: ${self.training_metrics['mae']:,.0f}")
            print(f"  • MAPE: {self.training_metrics['mape']:.1f}%")

        if self.validation_metrics:
            print(f"\nValidation Performance:")
            print(f"  • R²: {self.validation_metrics['r2']:.3f}")
            print(f"  • RMSE: ${self.validation_metrics['rmse']:,.0f}")
            print(f"  • MAE: ${self.validation_metrics['mae']:,.0f}")
            print(f"  • MAPE: {self.validation_metrics['mape']:.1f}%")

            # Overfitting check
            if self.training_metrics:
                r2_diff = self.training_metrics['r2'] - self.validation_metrics['r2']
                rmse_ratio = self.validation_metrics['rmse'] / self.training_metrics['rmse']

                print(f"\nOverfitting Analysis:")
                print(f"  • R² difference (train - val): {r2_diff:.3f}")
                print(f"  • RMSE ratio (val / train): {rmse_ratio:.3f}")

                if r2_diff > 0.1:
                    print(f"  ⚠️ Potential overfitting detected (R² diff > 0.1)")
                elif rmse_ratio > 1.2:
                    print(f"  ⚠️ Potential overfitting detected (RMSE ratio > 1.2)")
                else:
                    print(f"  ✅ No significant overfitting detected")

        if self.cv_scores:
            print(f"\nCross-Validation:")
            if 'mean_squared_error_mean' in self.cv_scores:
                cv_rmse = np.sqrt(self.cv_scores['mean_squared_error_mean'])
                cv_rmse_std = np.sqrt(self.cv_scores['mean_squared_error_std'])
                print(f"  • CV RMSE: ${cv_rmse:,.0f} ± ${cv_rmse_std:,.0f}")

            if 'r2_mean' in self.cv_scores:
                print(f"  • CV R²: {self.cv_scores['r2_mean']:.3f} ± {self.cv_scores['r2_std']:.3f}")

    def __str__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} (status: {status})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


# Test the base model functionality
if __name__ == "__main__":
    print("Testing Base Model functionality...")

    # Create a simple test implementation
    class TestModel(BaseModel):
        def _create_model(self, **kwargs):
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**kwargs)

        def __init__(self, **kwargs):
            super().__init__("Test Model", **kwargs)
            self.model = self._create_model(**kwargs)

    # Test with sample California housing data
    np.random.seed(42)
    X_sample = pd.DataFrame({
        'median_income': np.random.uniform(0.5, 15, 100),
        'total_rooms': np.random.uniform(500, 8000, 100),
        'housing_median_age': np.random.uniform(1, 52, 100)
    })
    y_sample = (X_sample['median_income'] * 30000 +
               X_sample['total_rooms'] * 20 +
               np.random.normal(0, 10000, 100))

    try:
        # Test model functionality
        test_model = TestModel()
        print(f"✅ Model created: {test_model}")

        # Test fitting
        test_model.fit(X_sample, y_sample)
        print(f"✅ Model fitted successfully")
        print(f"  • Feature count: {len(test_model.feature_names)}")
        print(f"  • Training R²: {test_model.training_metrics['r2']:.3f}")

        # Test predictions
        predictions = test_model.predict(X_sample)
        print(f"✅ Predictions generated: {len(predictions)} values")

        # Test cross-validation
        cv_scores = test_model.cross_validate(X_sample, y_sample, cv=3)
        print(f"✅ Cross-validation completed")

        print(f"\n✅ Base Model class working correctly!")

    except Exception as e:
        print(f"\n❌ Error in Base Model: {e}")
        import traceback
        traceback.print_exc()