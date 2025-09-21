"""
Model training and evaluation pipeline for California housing prediction.
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
    RandomForestModel, GradientBoostingModel, ExtraTreesModel
)

# Try to import XGBoost
try:
    from src.models.ensemble_models import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class CaliforniaHousingModelTrainer:
    """Comprehensive model training system for California housing prediction."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series):
        """
        Initialize model trainer for California housing.

        Args:
            X_train: Training features
            y_train: Training target (house values)
            X_val: Validation features
            y_val: Validation target (house values)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Model registry
        self.models = {}
        self.trained_models = {}
        self.model_results = {}

        # Hyperparameter grids optimized for California housing
        self.hyperparameter_grids = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
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
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0]
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

    def register_california_housing_models(self) -> None:
        """Register default set of models optimized for California housing."""
        print("📝 Registering California housing models...")

        # Linear models
        self.register_model(LinearRegressionModel, 'linear')
        self.register_model(RidgeRegressionModel, 'ridge', alpha=1.0)
        self.register_model(LassoRegressionModel, 'lasso', alpha=0.1)
        self.register_model(ElasticNetModel, 'elasticnet', alpha=0.1, l1_ratio=0.5)

        # Ensemble models
        self.register_model(RandomForestModel, 'random_forest',
                          n_estimators=100, max_depth=20, min_samples_split=5,
                          random_state=RANDOM_STATE)

        self.register_model(GradientBoostingModel, 'gradient_boosting',
                          n_estimators=100, learning_rate=0.1, max_depth=5,
                          random_state=RANDOM_STATE)

        self.register_model(ExtraTreesModel, 'extra_trees',
                          n_estimators=100, max_depth=20, min_samples_split=5,
                          random_state=RANDOM_STATE)

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.register_model(XGBoostModel, 'xgboost',
                              n_estimators=100, learning_rate=0.1, max_depth=6,
                              random_state=RANDOM_STATE)

        # Polynomial regression for non-linear relationships
        self.register_model(PolynomialRegressionModel, 'polynomial',
                          degree=2, alpha=10.0)

        print(f"✅ Registered {len(self.models)} models for California housing prediction")

    def train_single_model(self, model_name: str, tune_hyperparameters: bool = False,
                          cv_folds: int = 5) -> BaseModel:
        """
        Train a single model for California housing prediction.

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

        print(f"\n🎯 Training {model_name} for California housing...")

        if tune_hyperparameters and model_name in self.hyperparameter_grids:
            # Hyperparameter tuning
            print(f"🔍 Tuning hyperparameters for {model_name}...")
            best_model = self._tune_hyperparameters(model_class, model_name, base_kwargs, cv_folds)
        else:
            # Train with default parameters
            print(f"🔧 Training {model_name} with default parameters...")
            best_model = model_class(**base_kwargs)
            best_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        # Perform cross-validation
        print(f"🔄 Running cross-validation...")
        cv_scores = best_model.cross_validate(self.X_train, self.y_train, cv=cv_folds)

        # Store results
        self.trained_models[model_name] = best_model
        self.model_results[model_name] = {
            'model': best_model,
            'training_metrics': best_model.training_metrics,
            'validation_metrics': best_model.validation_metrics,
            'cv_scores': cv_scores
        }

        print(f"✅ {model_name} training completed successfully")
        return best_model

    def train_all_models(self, tune_hyperparameters: bool = False,
                        cv_folds: int = 5) -> Dict[str, BaseModel]:
        """
        Train all registered models for California housing.

        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of trained models
        """
        if not self.models:
            self.register_california_housing_models()

        print(f"🚀 TRAINING {len(self.models)} MODELS FOR CALIFORNIA HOUSING")
        print("="*70)

        training_summary = []

        for model_name in self.models.keys():
            try:
                start_time = datetime.now()
                model = self.train_single_model(model_name, tune_hyperparameters, cv_folds)
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()

                # Collect summary info
                val_r2 = model.validation_metrics.get('r2', 0) if model.validation_metrics else 0
                val_rmse = model.validation_metrics.get('rmse', 0) if model.validation_metrics else 0

                training_summary.append({
                    'model': model_name,
                    'training_time': training_time,
                    'validation_r2': val_r2,
                    'validation_rmse': val_rmse,
                    'status': 'success'
                })

            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                training_summary.append({
                    'model': model_name,
                    'training_time': 0,
                    'validation_r2': 0,
                    'validation_rmse': float('inf'),
                    'status': 'failed',
                    'error': str(e)
                })

        # Print training summary
        print(f"\n📊 TRAINING SUMMARY")
        print("="*70)
        successful_models = [s for s in training_summary if s['status'] == 'success']
        failed_models = [s for s in training_summary if s['status'] == 'failed']

        print(f"✅ Successful: {len(successful_models)}/{len(training_summary)} models")
        if failed_models:
            print(f"❌ Failed: {len(failed_models)} models")
            for failed in failed_models:
                print(f"  • {failed['model']}: {failed['error']}")

        print("\n✅ Model training completed!")
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
            Best model with optimal parameters
        """
        # Create base model for hyperparameter tuning
        base_model = model_class(**base_kwargs)
        param_grid = self.hyperparameter_grids[model_name]

        print(f"  🔍 Searching {len(param_grid)} hyperparameters...")

        # Use RandomizedSearchCV for faster tuning on larger parameter spaces
        if len(param_grid) > 3 or any(len(v) > 5 for v in param_grid.values()):
            search = RandomizedSearchCV(
                base_model.model,
                param_grid,
                n_iter=20,  # Limit iterations for faster training
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0,
                random_state=RANDOM_STATE
            )
        else:
            search = GridSearchCV(
                base_model.model,
                param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )

        # Fit search
        search.fit(self.X_train, self.y_train)

        # Create best model with optimal parameters
        best_params = {**base_kwargs, **search.best_params_}
        best_model = model_class(**best_params)
        best_model.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        print(f"  ✅ Best parameters: {search.best_params_}")
        print(f"  📊 Best CV score: ${np.sqrt(-search.best_score_):,.0f} RMSE")

        return best_model

    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all trained models and create comparison.

        Returns:
            DataFrame with model comparison
        """
        if not self.trained_models:
            raise ValueError("No models have been trained yet")

        print("📊 Evaluating all trained models...")

        evaluation_data = []

        for model_name, model in self.trained_models.items():
            # Get predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)

            # Calculate comprehensive metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            train_mae = mean_absolute_error(self.y_train, train_pred)
            val_mae = mean_absolute_error(self.y_val, val_pred)
            train_r2 = r2_score(self.y_train, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)

            # Get CV scores
            cv_rmse = model.cv_scores.get('mean_squared_error_mean', np.nan)
            cv_rmse = np.sqrt(cv_rmse) if not np.isnan(cv_rmse) else np.nan
            cv_r2 = model.cv_scores.get('r2_mean', np.nan)

            # Calculate additional metrics
            mean_house_value = self.y_val.mean()
            rmse_relative = val_rmse / mean_house_value * 100  # RMSE as % of mean house value

            evaluation_data.append({
                'model': model_name,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'cv_rmse': cv_rmse,
                'cv_r2': cv_r2,
                'rmse_relative_pct': rmse_relative,
                'overfit_ratio': val_rmse / train_rmse if train_rmse > 0 else np.inf,
                'r2_difference': train_r2 - val_r2
            })

        evaluation_df = pd.DataFrame(evaluation_data)
        evaluation_df = evaluation_df.sort_values('val_rmse')  # Sort by validation RMSE (lower is better)

        return evaluation_df

    def get_best_model(self, metric: str = 'val_rmse') -> Tuple[str, BaseModel]:
        """
        Get the best performing model for California housing.

        Args:
            metric: Metric to use for selection

        Returns:
            Tuple of (model_name, model)
        """
        evaluation_df = self.evaluate_models()

        if metric in ['val_rmse', 'train_rmse', 'cv_rmse', 'overfit_ratio', 'rmse_relative_pct']:
            # Lower is better
            best_model_name = evaluation_df.loc[evaluation_df[metric].idxmin(), 'model']
        else:  # For R² scores and other metrics where higher is better
            best_model_name = evaluation_df.loc[evaluation_df[metric].idxmax(), 'model']

        return best_model_name, self.trained_models[best_model_name]

    def plot_model_comparison(self, figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Create comprehensive model comparison visualizations.

        Args:
            figsize: Figure size
        """
        evaluation_df = self.evaluate_models()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Validation RMSE comparison
        axes[0,0].bar(evaluation_df['model'], evaluation_df['val_rmse'], color='lightcoral')
        axes[0,0].set_title('Validation RMSE Comparison')
        axes[0,0].set_ylabel('RMSE ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Add value labels
        for i, v in enumerate(evaluation_df['val_rmse']):
            axes[0,0].text(i, v + max(evaluation_df['val_rmse']) * 0.01, f'${v/1000:.0f}K',
                          ha='center', va='bottom', fontsize=9)

        # 2. Validation R² comparison
        axes[0,1].bar(evaluation_df['model'], evaluation_df['val_r2'], color='lightblue')
        axes[0,1].set_title('Validation R² Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 1)

        # Add value labels
        for i, v in enumerate(evaluation_df['val_r2']):
            axes[0,1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        # 3. Overfitting analysis (Train vs Validation RMSE)
        axes[1,0].scatter(evaluation_df['train_rmse'], evaluation_df['val_rmse'], s=100, alpha=0.7)
        for i, model in enumerate(evaluation_df['model']):
            axes[1,0].annotate(model,
                              (evaluation_df.iloc[i]['train_rmse'], evaluation_df.iloc[i]['val_rmse']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Perfect prediction line
        min_rmse = min(evaluation_df['train_rmse'].min(), evaluation_df['val_rmse'].min())
        max_rmse = max(evaluation_df['train_rmse'].max(), evaluation_df['val_rmse'].max())
        axes[1,0].plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'r--', alpha=0.8)
        axes[1,0].set_xlabel('Training RMSE ($)')
        axes[1,0].set_ylabel('Validation RMSE ($)')
        axes[1,0].set_title('Overfitting Analysis')

        # 4. Model performance ranking
        # Sort by validation R² (descending)
        sorted_df = evaluation_df.sort_values('val_r2', ascending=False)
        y_pos = np.arange(len(sorted_df))

        bars = axes[1,1].barh(y_pos, sorted_df['val_r2'], color='lightgreen')
        axes[1,1].set_yticks(y_pos)
        axes[1,1].set_yticklabels(sorted_df['model'])
        axes[1,1].set_xlabel('Validation R² Score')
        axes[1,1].set_title('Model Ranking by R² Score')
        axes[1,1].set_xlim(0, 1)

        # Add value labels
        for i, v in enumerate(sorted_df['val_r2']):
            axes[1,1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def save_models(self, models_dir: Path = None) -> None:
        """
        Save all trained models for California housing.

        Args:
            models_dir: Directory to save models
        """
        if models_dir is None:
            models_dir = MODELS_DIR

        models_dir.mkdir(exist_ok=True, parents=True)

        print(f"💾 Saving {len(self.trained_models)} trained models...")

        # Save individual models
        for model_name, model in self.trained_models.items():
            model_path = models_dir / f"{model_name}_california_housing.pkl"
            model.save_model(model_path)

        # Save evaluation results
        if self.trained_models:
            evaluation_df = self.evaluate_models()
            evaluation_df.to_csv(models_dir / "california_housing_model_evaluation.csv", index=False)

        # Save training summary
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'california_housing',
            'models_trained': list(self.trained_models.keys()),
            'training_data_shape': self.X_train.shape,
            'validation_data_shape': self.X_val.shape,
            'best_model': self.get_best_model()[0],
            'target_column': 'median_house_value'
        }

        joblib.dump(training_summary, models_dir / "california_housing_training_summary.pkl")
        print(f"✅ Models saved to: {models_dir}")

    def generate_training_report(self) -> str:
        """
        Generate comprehensive training report for California housing models.

        Returns:
            Training report string
        """
        if not self.trained_models:
            return "No models have been trained yet."

        evaluation_df = self.evaluate_models()
        best_model_name, best_model = self.get_best_model()

        report = []
        report.append("=" * 70)
        report.append("CALIFORNIA HOUSING MODEL TRAINING REPORT")
        report.append("=" * 70)

        # Training overview
        report.append(f"\nTRAINING OVERVIEW:")
        report.append(f"Dataset: California Housing")
        report.append(f"Models trained: {len(self.trained_models)}")
        report.append(f"Training samples: {self.X_train.shape[0]:,}")
        report.append(f"Validation samples: {self.X_val.shape[0]:,}")
        report.append(f"Features: {self.X_train.shape[1]}")
        report.append(f"Target: House values (median_house_value)")

        # Best model
        report.append(f"\n🏆 BEST MODEL: {best_model_name}")
        best_metrics = evaluation_df[evaluation_df['model'] == best_model_name].iloc[0]
        report.append(f"Validation RMSE: ${best_metrics['val_rmse']:,.0f}")
        report.append(f"Validation R²: {best_metrics['val_r2']:.4f}")
        report.append(f"Validation MAE: ${best_metrics['val_mae']:,.0f}")
        report.append(f"RMSE as % of mean house value: {best_metrics['rmse_relative_pct']:.1f}%")

        if not np.isnan(best_metrics['cv_rmse']):
            report.append(f"Cross-validation RMSE: ${best_metrics['cv_rmse']:,.0f}")

        # All models performance
        report.append(f"\nALL MODELS PERFORMANCE:")
        report.append(evaluation_df.to_string(index=False, float_format='%.4f'))

        # Feature importance (if available for best model)
        importance_df = best_model.get_feature_importance()
        if importance_df is not None:
            report.append(f"\nTOP 10 FEATURES ({best_model_name}):")
            for _, row in importance_df.head(10).iterrows():
                report.append(f"  {row['feature']:<30}: {row['importance']:.4f}")

        # Model recommendations
        report.append(f"\nMODEL RECOMMENDATIONS:")
        top_3_models = evaluation_df.head(3)
        for i, (_, row) in enumerate(top_3_models.iterrows(), 1):
            report.append(f"{i}. {row['model']} - R²: {row['val_r2']:.3f}, RMSE: ${row['val_rmse']:,.0f}")

        return "\n".join(report)


# Test the training pipeline
if __name__ == "__main__":
    print("Testing California Housing Model Trainer...")

    # Create sample California housing data
    np.random.seed(42)
    n_samples = 1000

    X_data = pd.DataFrame({
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'total_rooms': np.random.uniform(500, 8000, n_samples),
        'housing_median_age': np.random.uniform(1, 52, n_samples),
        'longitude': np.random.uniform(-124, -114, n_samples),
        'latitude': np.random.uniform(32, 42, n_samples),
        'population': np.random.uniform(300, 5000, n_samples),
        'households': np.random.uniform(100, 1800, n_samples)
    })

    # Create realistic target
    y_data = (
        X_data['median_income'] * 40000 +
        X_data['total_rooms'] * 30 +
        (50 - X_data['housing_median_age']) * 1000 +
        np.random.normal(0, 20000, n_samples)
    )
    y_data = pd.Series(np.clip(y_data, 50000, 500000), name='median_house_value')

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_data[:split_idx], X_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]

    try:
        # Test model trainer
        trainer = CaliforniaHousingModelTrainer(X_train, y_train, X_val, y_val)
        trainer.register_california_housing_models()

        # Train a subset of models for testing (faster)
        test_models = ['linear', 'ridge', 'random_forest']
        for model_name in test_models:
            if model_name in trainer.models:
                model = trainer.train_single_model(model_name, tune_hyperparameters=False)

        # Evaluate models
        if trainer.trained_models:
            evaluation_df = trainer.evaluate_models()
            best_model_name, best_model = trainer.get_best_model()

            print(f"\n📊 Evaluation Results:")
            print(evaluation_df[['model', 'val_r2', 'val_rmse']].round(4))

            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"Best R²: {evaluation_df.iloc[0]['val_r2']:.4f}")

            print(f"\n✅ Model Trainer working correctly!")

    except Exception as e:
        print(f"\n❌ Error in Model Trainer: {e}")
        import traceback
        traceback.print_exc()