"""
Model validation and analysis utilities for California housing prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.base_model import BaseModel


class CaliforniaHousingModelValidator:
    """Comprehensive model validation for California housing prediction."""

    def __init__(self, models: Dict[str, BaseModel], X_train: pd.DataFrame,
                 y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Initialize model validator for California housing.

        Args:
            models: Dictionary of trained models
            X_train: Training features
            y_train: Training target (house values)
            X_val: Validation features
            y_val: Validation target (house values)
        """
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Calculate some dataset statistics for context
        self.mean_house_value = y_train.mean()
        self.house_value_std = y_train.std()
        self.house_value_range = y_train.max() - y_train.min()

    def validate_single_model(self, model: BaseModel, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive validation for a single California housing model.

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

        # Comprehensive metrics
        results['train_metrics'] = {
            'rmse': float(np.sqrt(mean_squared_error(self.y_train, train_pred))),
            'mae': float(mean_absolute_error(self.y_train, train_pred)),
            'r2': float(r2_score(self.y_train, train_pred)),
            'mape': float(np.mean(np.abs((self.y_train - train_pred) / self.y_train)) * 100)
        }

        results['val_metrics'] = {
            'rmse': float(np.sqrt(mean_squared_error(self.y_val, val_pred))),
            'mae': float(mean_absolute_error(self.y_val, val_pred)),
            'r2': float(r2_score(self.y_val, val_pred)),
            'mape': float(np.mean(np.abs((self.y_val - val_pred) / self.y_val)) * 100)
        }

        # Housing-specific metrics
        results['housing_metrics'] = {
            'rmse_as_pct_of_mean_price': results['val_metrics']['rmse'] / self.mean_house_value * 100,
            'rmse_as_pct_of_price_range': results['val_metrics']['rmse'] / self.house_value_range * 100,
            'prediction_accuracy_within_10k': float(np.mean(np.abs(self.y_val - val_pred) <= 10000) * 100),
            'prediction_accuracy_within_20k': float(np.mean(np.abs(self.y_val - val_pred) <= 20000) * 100),
            'prediction_accuracy_within_50k': float(np.mean(np.abs(self.y_val - val_pred) <= 50000) * 100)
        }

        # Cross-validation
        try:
            cv_scores = cross_val_score(model.model, self.X_train, self.y_train,
                                       cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_r2_scores = cross_val_score(model.model, self.X_train, self.y_train,
                                          cv=cv_folds, scoring='r2', n_jobs=-1)

            results['cv_metrics'] = {
                'cv_rmse_mean': float(np.sqrt(-cv_scores.mean())),
                'cv_rmse_std': float(np.sqrt(cv_scores.std())),
                'cv_r2_mean': float(cv_r2_scores.mean()),
                'cv_r2_std': float(cv_r2_scores.std()),
                'cv_scores': cv_scores.tolist()
            }
        except Exception as e:
            print(f"⚠️ Cross-validation failed for {model.name}: {e}")
            results['cv_metrics'] = {}

        # Overfitting analysis
        results['overfitting'] = {
            'rmse_ratio': results['val_metrics']['rmse'] / results['train_metrics']['rmse'],
            'r2_difference': results['train_metrics']['r2'] - results['val_metrics']['r2'],
            'is_overfitting': results['val_metrics']['rmse'] / results['train_metrics']['rmse'] > 1.15
        }

        # Residual analysis
        train_residuals = self.y_train - train_pred
        val_residuals = self.y_val - val_pred

        results['residual_analysis'] = {
            'train_residual_mean': float(train_residuals.mean()),
            'train_residual_std': float(train_residuals.std()),
            'val_residual_mean': float(val_residuals.mean()),
            'val_residual_std': float(val_residuals.std()),
            'residual_skewness': float(val_residuals.skew()),
            'residual_kurtosis': float(val_residuals.kurtosis())
        }

        # Prediction quality analysis
        val_errors = np.abs(self.y_val - val_pred)
        results['prediction_quality'] = {
            'mean_absolute_error': float(val_errors.mean()),
            'median_absolute_error': float(val_errors.median()),
            'max_absolute_error': float(val_errors.max()),
            'error_percentiles': {
                '90th': float(np.percentile(val_errors, 90)),
                '95th': float(np.percentile(val_errors, 95)),
                '99th': float(np.percentile(val_errors, 99))
            }
        }

        return results

    def compare_models(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare all models across multiple metrics.

        Args:
            metrics: Metrics to include in comparison

        Returns:
            Comparison DataFrame sorted by validation RMSE
        """
        if metrics is None:
            metrics = ['val_rmse', 'val_r2', 'val_mae', 'cv_rmse_mean', 'rmse_ratio',
                      'rmse_as_pct_of_mean_price', 'prediction_accuracy_within_20k']

        comparison_data = []

        for model_name, model in self.models.items():
            validation_results = self.validate_single_model(model)

            row = {'model': model_name}

            # Extract requested metrics
            for metric in metrics:
                if metric.startswith('val_'):
                    metric_key = metric.replace('val_', '')
                    row[metric] = validation_results['val_metrics'].get(metric_key, np.nan)
                elif metric.startswith('train_'):
                    metric_key = metric.replace('train_', '')
                    row[metric] = validation_results['train_metrics'].get(metric_key, np.nan)
                elif metric.startswith('cv_'):
                    row[metric] = validation_results['cv_metrics'].get(metric, np.nan)
                elif metric in validation_results['overfitting']:
                    row[metric] = validation_results['overfitting'][metric]
                elif metric in validation_results['housing_metrics']:
                    row[metric] = validation_results['housing_metrics'][metric]
                else:
                    row[metric] = np.nan

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('val_rmse')

    def plot_model_performance(self, figsize: Tuple[int, int] = (18, 15)) -> None:
        """
        Create comprehensive performance plots for California housing models.

        Args:
            figsize: Figure size
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        # Get validation results for all models
        all_results = {}
        for model_name, model in self.models.items():
            all_results[model_name] = self.validate_single_model(model)

        models = list(self.models.keys())

        # 1. Validation RMSE comparison
        val_rmse = [all_results[model]['val_metrics']['rmse'] for model in models]
        bars1 = axes[0].bar(models, val_rmse, color='lightcoral')
        axes[0].set_title('Validation RMSE Comparison')
        axes[0].set_ylabel('RMSE ($)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # Add value labels
        for bar, value in zip(bars1, val_rmse):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(val_rmse)*0.01,
                        f'${value/1000:.0f}K', ha='center', va='bottom', fontsize=8)

        # 2. Validation R² comparison
        val_r2 = [all_results[model]['val_metrics']['r2'] for model in models]
        bars2 = axes[1].bar(models, val_r2, color='lightblue')
        axes[1].set_title('Validation R² Comparison')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1)

        # Add value labels
        for bar, value in zip(bars2, val_r2):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # 3. RMSE as percentage of mean house value
        rmse_pct = [all_results[model]['housing_metrics']['rmse_as_pct_of_mean_price'] for model in models]
        axes[2].bar(models, rmse_pct, color='lightgreen')
        axes[2].set_title('RMSE as % of Mean House Value')
        axes[2].set_ylabel('RMSE (%)')
        axes[2].tick_params(axis='x', rotation=45)

        # 4. Overfitting analysis
        train_rmse = [all_results[model]['train_metrics']['rmse'] for model in models]
        val_rmse = [all_results[model]['val_metrics']['rmse'] for model in models]

        axes[3].scatter(train_rmse, val_rmse, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[3].annotate(model, (train_rmse[i], val_rmse[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        min_rmse = min(min(train_rmse), min(val_rmse))
        max_rmse = max(max(train_rmse), max(val_rmse))
        axes[3].plot([min_rmse, max_rmse], [min_rmse, max_rmse], 'r--', alpha=0.8)
        axes[3].set_xlabel('Training RMSE ($)')
        axes[3].set_ylabel('Validation RMSE ($)')
        axes[3].set_title('Overfitting Analysis')

        # 5. Prediction accuracy within thresholds
        accuracy_20k = [all_results[model]['housing_metrics']['prediction_accuracy_within_20k'] for model in models]
        axes[4].bar(models, accuracy_20k, color='gold')
        axes[4].set_title('Predictions within $20K (Accuracy %)')
        axes[4].set_ylabel('Accuracy (%)')
        axes[4].tick_params(axis='x', rotation=45)

        # 6-8. Individual model predictions vs actual (show top 3 models)
        sorted_models = sorted(models, key=lambda m: all_results[m]['val_metrics']['rmse'])

        for i, model_name in enumerate(sorted_models[:3]):
            ax_idx = 5 + i
            model = self.models[model_name]
            val_pred = model.predict(self.X_val)

            axes[ax_idx].scatter(self.y_val, val_pred, alpha=0.6, s=20)
            axes[ax_idx].plot([self.y_val.min(), self.y_val.max()],
                             [self.y_val.min(), self.y_val.max()], 'r--', lw=2)
            axes[ax_idx].set_xlabel('Actual House Value ($)')
            axes[ax_idx].set_ylabel('Predicted House Value ($)')
            axes[ax_idx].set_title(f'{model_name} - Predictions vs Actual')

            # Format axes
            axes[ax_idx].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            axes[ax_idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

            # Add R² to plot
            r2 = r2_score(self.y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
            axes[ax_idx].text(0.05, 0.95, f'R² = {r2:.3f}\\nRMSE = ${rmse/1000:.0f}K',
                             transform=axes[ax_idx].transAxes,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                             verticalalignment='top', fontsize=9)

        # Hide unused subplots
        for i in range(len(sorted_models) + 5, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_residual_analysis(self, model_name: str, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot detailed residual analysis for California housing prediction.

        Args:
            model_name: Name of the model to analyze
            figsize: Figure size
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        validation_results = self.validate_single_model(model)

        val_pred = model.predict(self.X_val)
        residuals = self.y_val - val_pred

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 1. Residuals vs Predicted
        axes[0,0].scatter(val_pred, residuals, alpha=0.6, s=20)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_xlabel('Predicted House Value ($)')
        axes[0,0].set_ylabel('Residuals ($)')
        axes[0,0].set_title('Residuals vs Predicted')
        axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # 2. Residuals distribution
        axes[0,1].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='lightblue')
        axes[0,1].axvline(residuals.mean(), color='red', linestyle='--', alpha=0.7)
        axes[0,1].set_xlabel('Residuals ($)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Residuals Distribution')

        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0,2])
        axes[0,2].set_title('Q-Q Plot (Normal)')

        # 4. Residuals vs actual values
        axes[1,0].scatter(self.y_val, residuals, alpha=0.6, s=20)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Actual House Value ($)')
        axes[1,0].set_ylabel('Residuals ($)')
        axes[1,0].set_title('Residuals vs Actual')
        axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # 5. Absolute residuals vs predicted (heteroscedasticity check)
        abs_residuals = np.abs(residuals)
        axes[1,1].scatter(val_pred, abs_residuals, alpha=0.6, s=20)
        axes[1,1].set_xlabel('Predicted House Value ($)')
        axes[1,1].set_ylabel('Absolute Residuals ($)')
        axes[1,1].set_title('Heteroscedasticity Check')
        axes[1,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

        # 6. Prediction error by house value range
        # Create price bins
        price_bins = pd.qcut(self.y_val, q=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        error_by_price = pd.DataFrame({'price_range': price_bins, 'abs_error': abs_residuals})

        # Box plot of errors by price range
        sns.boxplot(data=error_by_price, x='price_range', y='abs_error', ax=axes[1,2])
        axes[1,2].set_title('Prediction Error by House Value Range')
        axes[1,2].set_xlabel('House Value Range')
        axes[1,2].set_ylabel('Absolute Error ($)')
        axes[1,2].tick_params(axis='x', rotation=45)

        plt.suptitle(f'Comprehensive Residual Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Print residual statistics
        print(f"📊 RESIDUAL STATISTICS - {model_name}")
        print("="*50)
        print(f"Mean residual: ${residuals.mean():,.0f}")
        print(f"Std residual: ${residuals.std():,.0f}")
        print(f"Mean absolute error: ${abs_residuals.mean():,.0f}")
        print(f"Median absolute error: ${abs_residuals.median():,.0f}")
        print(f"Max absolute error: ${abs_residuals.max():,.0f}")
        print(f"Residual skewness: {residuals.skew():.3f}")

    def plot_learning_curves(self, model_name: str, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot learning curves for California housing model.

        Args:
            model_name: Name of the model to analyze
            figsize: Figure size
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        print(f"📈 Generating learning curves for {model_name}...")

        try:
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
            plt.ylabel('RMSE ($)')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            plt.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            plt.show()

        except Exception as e:
            print(f"❌ Error generating learning curves: {e}")

    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report for California housing models.

        Returns:
            Validation report string
        """
        comparison_df = self.compare_models()

        report = []
        report.append("=" * 70)
        report.append("CALIFORNIA HOUSING MODEL VALIDATION REPORT")
        report.append("=" * 70)

        # Dataset context
        report.append(f"\nDATASET CONTEXT:")
        report.append(f"Mean house value: ${self.mean_house_value:,.0f}")
        report.append(f"House value std: ${self.house_value_std:,.0f}")
        report.append(f"House value range: ${self.house_value_range:,.0f}")
        report.append(f"Training samples: {len(self.X_train):,}")
        report.append(f"Validation samples: {len(self.X_val):,}")

        # Model ranking
        report.append(f"\nMODEL RANKING (by Validation RMSE):")
        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            rmse_pct = row.get('rmse_as_pct_of_mean_price', 0)
            report.append(f"{i}. {row['model']}: ${row['val_rmse']:,.0f} ({rmse_pct:.1f}% of mean price)")

        # Best model analysis
        best_model_name = comparison_df.iloc[0]['model']
        best_model = self.models[best_model_name]
        best_results = self.validate_single_model(best_model)

        report.append(f"\n🏆 BEST MODEL ANALYSIS: {best_model_name}")
        report.append(f"Validation RMSE: ${best_results['val_metrics']['rmse']:,.0f}")
        report.append(f"Validation R²: {best_results['val_metrics']['r2']:.4f}")
        report.append(f"Validation MAE: ${best_results['val_metrics']['mae']:,.0f}")
        report.append(f"RMSE as % of mean price: {best_results['housing_metrics']['rmse_as_pct_of_mean_price']:.1f}%")

        if best_results['cv_metrics']:
            report.append(f"Cross-validation RMSE: ${best_results['cv_metrics']['cv_rmse_mean']:,.0f} ± ${best_results['cv_metrics']['cv_rmse_std']:,.0f}")

        # Prediction accuracy
        housing_metrics = best_results['housing_metrics']
        report.append(f"\nPREDICTION ACCURACY:")
        report.append(f"Within $10K: {housing_metrics['prediction_accuracy_within_10k']:.1f}%")
        report.append(f"Within $20K: {housing_metrics['prediction_accuracy_within_20k']:.1f}%")
        report.append(f"Within $50K: {housing_metrics['prediction_accuracy_within_50k']:.1f}%")

        # Overfitting analysis
        overfitting = best_results['overfitting']
        report.append(f"\nOVERFITTING ANALYSIS:")
        report.append(f"RMSE Ratio (Val/Train): {overfitting['rmse_ratio']:.3f}")
        report.append(f"R² Difference (Train-Val): {overfitting['r2_difference']:.3f}")
        report.append(f"Overfitting detected: {'Yes' if overfitting['is_overfitting'] else 'No'}")

        # Feature importance (if available)
        importance_df = best_model.get_feature_importance()
        if importance_df is not None:
            report.append(f"\nTOP 10 FEATURES ({best_model_name}):")
            for _, row in importance_df.head(10).iterrows():
                report.append(f"  {row['feature']:<30}: {row['importance']:.4f}")

        # Full comparison table
        report.append(f"\nFULL MODEL COMPARISON:")
        # Select key columns for the report
        key_columns = ['model', 'val_rmse', 'val_r2', 'val_mae', 'rmse_ratio']
        available_columns = [col for col in key_columns if col in comparison_df.columns]
        report.append(comparison_df[available_columns].to_string(index=False, float_format='%.4f'))

        return "\n".join(report)

    def create_prediction_confidence_analysis(self, model_name: str) -> Dict[str, Any]:
        """
        Analyze prediction confidence for different house value ranges.

        Args:
            model_name: Name of model to analyze

        Returns:
            Confidence analysis results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        val_pred = model.predict(self.X_val)
        errors = np.abs(self.y_val - val_pred)

        # Create house value bins
        price_bins = pd.qcut(self.y_val, q=5, labels=['Low', 'Low-Medium', 'Medium', 'Medium-High', 'High'])

        confidence_analysis = {}
        for bin_name in price_bins.cat.categories:
            bin_mask = price_bins == bin_name
            bin_errors = errors[bin_mask]
            bin_prices = self.y_val[bin_mask]

            if len(bin_errors) > 0:
                confidence_analysis[bin_name] = {
                    'sample_count': len(bin_errors),
                    'mean_error': float(bin_errors.mean()),
                    'median_error': float(bin_errors.median()),
                    'error_std': float(bin_errors.std()),
                    'mean_price': float(bin_prices.mean()),
                    'price_range': f"${bin_prices.min():,.0f} - ${bin_prices.max():,.0f}",
                    'accuracy_within_10k': float((bin_errors <= 10000).mean() * 100),
                    'accuracy_within_20k': float((bin_errors <= 20000).mean() * 100)
                }

        return confidence_analysis


# Test the validation system
if __name__ == "__main__":
    print("Testing California Housing Model Validator...")

    # Create sample models and data for testing
    from src.models.linear_models import LinearRegressionModel, RidgeRegressionModel
    from src.models.ensemble_models import RandomForestModel

    # Sample data
    np.random.seed(42)
    n_samples = 500

    X_data = pd.DataFrame({
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'total_rooms': np.random.uniform(500, 8000, n_samples),
        'housing_median_age': np.random.uniform(1, 52, n_samples)
    })

    y_data = (
        X_data['median_income'] * 40000 +
        X_data['total_rooms'] * 30 +
        np.random.normal(0, 20000, n_samples)
    )
    y_data = pd.Series(np.clip(y_data, 50000, 500000), name='median_house_value')

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_data[:split_idx], X_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]

    # Train sample models
    models = {
        'linear': LinearRegressionModel(),
        'ridge': RidgeRegressionModel(alpha=1.0),
        'random_forest': RandomForestModel(n_estimators=50, random_state=42)
    }

    for model in models.values():
        model.fit(X_train, y_train, X_val, y_val)

    try:
        # Test validator
        validator = CaliforniaHousingModelValidator(models, X_train, y_train, X_val, y_val)

        # Test single model validation
        validation_results = validator.validate_single_model(models['linear'])
        print(f"✅ Single model validation working")

        # Test model comparison
        comparison_df = validator.compare_models()
        print(f"✅ Model comparison working: {len(comparison_df)} models compared")

        # Test confidence analysis
        confidence_analysis = validator.create_prediction_confidence_analysis('linear')
        print(f"✅ Confidence analysis working: {len(confidence_analysis)} price ranges analyzed")

        print(f"\n✅ Model Validator working correctly!")

    except Exception as e:
        print(f"\n❌ Error in Model Validator: {e}")
        import traceback
        traceback.print_exc()