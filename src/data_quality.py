"""
Data quality assessment utilities for house price prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings


class DataQualityAssessor:
    """Assess data quality issues and provide actionable recommendations."""

    def __init__(self, data: pd.DataFrame, name: str = "Dataset"):
        """
        Initialize data quality assessor.

        Args:
            data: Dataset to assess
            name: Name of the dataset for reporting
        """
        self.data = data.copy()
        self.name = name
        self.original_shape = data.shape

    def assess_missing_values(self) -> Dict[str, Any]:
        """
        Comprehensive missing value assessment.

        Returns:
            Dictionary with missing value analysis and recommendations
        """
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100

        # Categorize missing value severity
        high_missing = missing_percentages[missing_percentages > 50].index.tolist()
        medium_missing = missing_percentages[(missing_percentages > 20) & (missing_percentages <= 50)].index.tolist()
        low_missing = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 20)].index.tolist()
        complete = missing_percentages[missing_percentages == 0].index.tolist()

        # Pattern analysis
        missing_patterns = self._analyze_missing_patterns()

        return {
            'summary': {
                'total_missing_values': int(missing_counts.sum()),
                'missing_percentage_overall': float(missing_counts.sum() / (len(self.data) * len(self.data.columns)) * 100),
                'columns_with_missing': int((missing_counts > 0).sum()),
                'complete_columns': int((missing_counts == 0).sum())
            },
            'by_severity': {
                'high_missing_columns': high_missing,
                'medium_missing_columns': medium_missing,
                'low_missing_columns': low_missing,
                'complete_columns': complete
            },
            'detailed_missing': {
                col: {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col])
                }
                for col in missing_counts[missing_counts > 0].index
            },
            'patterns': missing_patterns,
            'recommendations': self._get_missing_value_recommendations(high_missing, medium_missing, low_missing)
        }

    def _analyze_missing_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in missing data.

        Returns:
            Dictionary with missing pattern analysis
        """
        # Rows with any missing values
        rows_with_missing = self.data.isnull().any(axis=1).sum()
        rows_complete = len(self.data) - rows_with_missing

        # Most common missing patterns
        missing_pattern_counts = self.data.isnull().value_counts().head(10)

        # Correlation between missing values
        missing_corr = self.data.isnull().corr()
        high_missing_corr = []

        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation threshold
                    high_missing_corr.append({
                        'feature_1': missing_corr.columns[i],
                        'feature_2': missing_corr.columns[j],
                        'correlation': float(corr_val)
                    })

        return {
            'rows_with_missing': int(rows_with_missing),
            'rows_complete': int(rows_complete),
            'complete_rows_percentage': float(rows_complete / len(self.data) * 100),
            'common_patterns': missing_pattern_counts.head(5).to_dict(),
            'highly_correlated_missing': high_missing_corr
        }

    def assess_data_types(self) -> Dict[str, Any]:
        """
        Assess appropriateness of data types.

        Returns:
            Dictionary with data type assessment and recommendations
        """
        type_issues = []
        recommendations = []

        for col in self.data.columns:
            current_dtype = str(self.data[col].dtype)

            # Check for potential numeric columns stored as objects
            if self.data[col].dtype == 'object':
                # Try to convert to numeric
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    numeric_converted = pd.to_numeric(self.data[col], errors='coerce')

                non_null_original = self.data[col].notna().sum()
                non_null_converted = numeric_converted.notna().sum()

                if non_null_original > 0:
                    conversion_success_rate = non_null_converted / non_null_original

                    # If most values can be converted to numeric (>80%), flag as potential issue
                    if conversion_success_rate > 0.8:
                        type_issues.append({
                            'column': col,
                            'current_type': current_dtype,
                            'suggested_type': 'numeric',
                            'conversion_success_rate': float(conversion_success_rate),
                            'issue_type': 'numeric_stored_as_object'
                        })
                        recommendations.append(f"Convert '{col}' to numeric type")

            # Check for potential categorical columns with few unique values
            elif self.data[col].dtype in ['int64', 'float64']:
                unique_count = self.data[col].nunique()
                total_count = len(self.data)

                # If few unique values relative to total, might be categorical
                if unique_count <= 20 and unique_count / total_count < 0.05:
                    type_issues.append({
                        'column': col,
                        'current_type': current_dtype,
                        'suggested_type': 'categorical',
                        'unique_count': int(unique_count),
                        'issue_type': 'potential_categorical'
                    })
                    recommendations.append(f"Consider treating '{col}' as categorical")

        return {
            'data_types_summary': self.data.dtypes.value_counts().to_dict(),
            'type_issues': type_issues,
            'recommendations': recommendations,
            'total_issues': len(type_issues)
        }

    def assess_outliers(self, numerical_columns: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive outlier assessment using multiple methods.

        Args:
            numerical_columns: List of columns to check (default: all numerical)

        Returns:
            Dictionary with outlier assessment
        """
        if numerical_columns is None:
            numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        outlier_analysis = {}

        for col in numerical_columns:
            if col in self.data.columns and self.data[col].notna().sum() > 0:
                data_col = self.data[col].dropna()

                # IQR method
                Q1 = data_col.quantile(0.25)
                Q3 = data_col.quantile(0.75)
                IQR = Q3 - Q1
                iqr_lower_bound = Q1 - 1.5 * IQR
                iqr_upper_bound = Q3 + 1.5 * IQR

                iqr_outliers = self.data[(self.data[col] < iqr_lower_bound) | (self.data[col] > iqr_upper_bound)]

                # Z-score method (for comparison)
                mean_val = data_col.mean()
                std_val = data_col.std()
                if std_val > 0:
                    z_scores = np.abs((self.data[col] - mean_val) / std_val)
                    z_outliers = self.data[z_scores > 3]  # 3 standard deviations
                    z_outlier_count = len(z_outliers)
                else:
                    z_outlier_count = 0

                # Modified Z-score using median
                median_val = data_col.median()
                mad = np.median(np.abs(data_col - median_val))
                if mad > 0:
                    modified_z_scores = 0.6745 * (self.data[col] - median_val) / mad
                    modified_z_outliers = self.data[np.abs(modified_z_scores) > 3.5]
                    modified_z_outlier_count = len(modified_z_outliers)
                else:
                    modified_z_outlier_count = 0

                outlier_analysis[col] = {
                    'iqr_method': {
                        'outlier_count': len(iqr_outliers),
                        'outlier_percentage': float(len(iqr_outliers) / len(self.data) * 100),
                        'lower_bound': float(iqr_lower_bound),
                        'upper_bound': float(iqr_upper_bound),
                        'outlier_indices': iqr_outliers.index.tolist()
                    },
                    'z_score_method': {
                        'outlier_count': int(z_outlier_count),
                        'outlier_percentage': float(z_outlier_count / len(self.data) * 100)
                    },
                    'modified_z_score_method': {
                        'outlier_count': int(modified_z_outlier_count),
                        'outlier_percentage': float(modified_z_outlier_count / len(self.data) * 100)
                    },
                    'statistics': {
                        'mean': float(mean_val),
                        'median': float(median_val),
                        'std': float(std_val),
                        'min': float(data_col.min()),
                        'max': float(data_col.max()),
                        'range': float(data_col.max() - data_col.min())
                    }
                }

        return outlier_analysis

    def assess_duplicates(self) -> Dict[str, Any]:
        """
        Assess duplicate records in the dataset.

        Returns:
            Dictionary with duplicate assessment
        """
        # Full row duplicates
        full_duplicates = self.data.duplicated().sum()
        full_duplicate_indices = self.data[self.data.duplicated()].index.tolist()

        # Duplicates excluding ID columns
        id_like_columns = [col for col in self.data.columns
                          if col.lower() in ['id', 'index'] or col.lower().endswith('_id')]

        columns_to_check = [col for col in self.data.columns if col not in id_like_columns]

        if columns_to_check:
            feature_duplicates = self.data[columns_to_check].duplicated().sum()
            feature_duplicate_indices = self.data[self.data[columns_to_check].duplicated()].index.tolist()
        else:
            feature_duplicates = 0
            feature_duplicate_indices = []

        # Potential duplicates with minor differences
        potential_duplicates = self._find_potential_duplicates(columns_to_check)

        return {
            'full_duplicates': {
                'count': int(full_duplicates),
                'percentage': float(full_duplicates / len(self.data) * 100),
                'indices': full_duplicate_indices
            },
            'feature_duplicates': {
                'count': int(feature_duplicates),
                'percentage': float(feature_duplicates / len(self.data) * 100),
                'indices': feature_duplicate_indices,
                'excluded_columns': id_like_columns
            },
            'potential_duplicates': potential_duplicates,
            'recommendations': self._get_duplicate_recommendations(full_duplicates, feature_duplicates)
        }

    def _find_potential_duplicates(self, columns: List[str], threshold: float = 0.9) -> Dict[str, Any]:
        """
        Find potential duplicates using similarity measures.

        Args:
            columns: Columns to consider for similarity
            threshold: Similarity threshold (0-1)

        Returns:
            Dictionary with potential duplicate information
        """
        # This is a simplified approach - for text data, you might use more sophisticated methods
        potential_duplicates = []

        # For numerical data, we can use correlation or distance measures
        numerical_cols = [col for col in columns if self.data[col].dtype in ['int64', 'float64']]

        if len(numerical_cols) > 1 and len(self.data) < 10000:  # Avoid expensive computation on large datasets
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                # Compute similarity matrix for a sample
                sample_size = min(1000, len(self.data))
                sample_data = self.data[numerical_cols].sample(sample_size, random_state=42).fillna(0)

                similarity_matrix = cosine_similarity(sample_data)

                # Find pairs with high similarity
                high_similarity_pairs = []
                for i in range(len(similarity_matrix)):
                    for j in range(i+1, len(similarity_matrix)):
                        if similarity_matrix[i][j] > threshold:
                            high_similarity_pairs.append({
                                'index_1': int(sample_data.index[i]),
                                'index_2': int(sample_data.index[j]),
                                'similarity': float(similarity_matrix[i][j])
                            })

                potential_duplicates = high_similarity_pairs[:10]  # Limit to top 10

            except ImportError:
                # sklearn not available for similarity computation
                pass

        return {
            'count': len(potential_duplicates),
            'pairs': potential_duplicates,
            'method': 'cosine_similarity',
            'threshold_used': threshold
        }

    def assess_cardinality(self) -> Dict[str, Any]:
        """
        Assess feature cardinality and identify potential issues.

        Returns:
            Dictionary with cardinality assessment
        """
        cardinality_analysis = {}

        for col in self.data.columns:
            unique_count = self.data[col].nunique()
            total_count = len(self.data)
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0

            # Categorize cardinality
            if cardinality_ratio == 1.0:
                category = 'unique_identifier'
            elif cardinality_ratio > 0.95:
                category = 'high_cardinality'
            elif cardinality_ratio < 0.01:
                category = 'low_cardinality'
            elif cardinality_ratio < 0.05:
                category = 'low_moderate_cardinality'
            else:
                category = 'normal_cardinality'

            cardinality_analysis[col] = {
                'unique_count': int(unique_count),
                'total_count': int(total_count),
                'cardinality_ratio': float(cardinality_ratio),
                'category': category,
                'data_type': str(self.data[col].dtype)
            }

        # Summarize by category
        category_summary = {}
        for col_info in cardinality_analysis.values():
            category = col_info['category']
            category_summary[category] = category_summary.get(category, 0) + 1

        return {
            'by_column': cardinality_analysis,
            'summary': category_summary,
            'recommendations': self._get_cardinality_recommendations(cardinality_analysis)
        }

    def assess_data_consistency(self) -> Dict[str, Any]:
        """
        Assess data consistency and format issues.

        Returns:
            Dictionary with consistency assessment
        """
        consistency_issues = []

        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Check for inconsistent formatting
                non_null_values = self.data[col].dropna()

                if len(non_null_values) > 0:
                    # Check for leading/trailing whitespace
                    whitespace_issues = (non_null_values != non_null_values.str.strip()).sum()

                    # Check for case inconsistencies
                    unique_values = non_null_values.unique()
                    case_variants = {}
                    for val in unique_values:
                        lower_val = str(val).lower()
                        if lower_val in case_variants:
                            case_variants[lower_val].append(val)
                        else:
                            case_variants[lower_val] = [val]

                    case_inconsistencies = {k: v for k, v in case_variants.items() if len(v) > 1}

                    if whitespace_issues > 0 or case_inconsistencies:
                        consistency_issues.append({
                            'column': col,
                            'whitespace_issues': int(whitespace_issues),
                            'case_inconsistencies': case_inconsistencies,
                            'total_unique_values': len(unique_values)
                        })

        return {
            'consistency_issues': consistency_issues,
            'columns_with_issues': len(consistency_issues),
            'recommendations': self._get_consistency_recommendations(consistency_issues)
        }

    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Returns:
            Complete data quality assessment
        """
        print(f"Generating data quality report for {self.name}...")

        report = {
            'metadata': {
                'dataset_name': self.name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'original_shape': self.original_shape,
                'current_shape': self.data.shape
            }
        }

        try:
            print("Assessing missing values...")
            report['missing_values'] = self.assess_missing_values()

            print("Assessing data types...")
            report['data_types'] = self.assess_data_types()

            print("Assessing outliers...")
            report['outliers'] = self.assess_outliers()

            print("Assessing duplicates...")
            report['duplicates'] = self.assess_duplicates()

            print("Assessing cardinality...")
            report['cardinality'] = self.assess_cardinality()

            print("Assessing data consistency...")
            report['consistency'] = self.assess_data_consistency()

            # Overall quality score
            report['overall_quality'] = self._calculate_quality_score(report)

            print("✅ Data quality report generation completed")

        except Exception as e:
            print(f"❌ Error during quality assessment: {e}")
            report['error'] = str(e)

        return report

    def _calculate_quality_score(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall data quality score.

        Args:
            report: Quality assessment report

        Returns:
            Dictionary with quality scores and grades
        """
        scores = {}

        # Missing values score (0-100)
        missing_pct = report['missing_values']['summary']['missing_percentage_overall']
        missing_score = max(0, 100 - missing_pct * 2)  # Penalize missing values
        scores['missing_values'] = missing_score

        # Duplicates score (0-100)
        duplicate_pct = report['duplicates']['full_duplicates']['percentage']
        duplicate_score = max(0, 100 - duplicate_pct * 5)  # Heavily penalize duplicates
        scores['duplicates'] = duplicate_score

        # Consistency score (0-100)
        consistency_issues = len(report['consistency']['consistency_issues'])
        total_text_columns = len([col for col in self.data.columns if self.data[col].dtype == 'object'])
        if total_text_columns > 0:
            consistency_score = max(0, 100 - (consistency_issues / total_text_columns) * 50)
        else:
            consistency_score = 100
        scores['consistency'] = consistency_score

        # Overall score
        overall_score = np.mean(list(scores.values()))

        # Grade assignment
        if overall_score >= 90:
            grade = 'A'
        elif overall_score >= 80:
            grade = 'B'
        elif overall_score >= 70:
            grade = 'C'
        elif overall_score >= 60:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'scores': scores,
            'overall_score': float(overall_score),
            'grade': grade,
            'interpretation': self._interpret_quality_score(overall_score)
        }

    def _interpret_quality_score(self, score: float) -> str:
        """Interpret the quality score."""
        if score >= 90:
            return "Excellent data quality. Ready for analysis."
        elif score >= 80:
            return "Good data quality. Minor issues to address."
        elif score >= 70:
            return "Fair data quality. Some cleaning recommended."
        elif score >= 60:
            return "Poor data quality. Significant cleaning required."
        else:
            return "Very poor data quality. Extensive cleaning needed."

    def _get_missing_value_recommendations(self, high: List[str], medium: List[str], low: List[str]) -> List[str]:
        """Generate recommendations for handling missing values."""
        recommendations = []

        if high:
            recommendations.append(f"Consider dropping columns with >50% missing data: {high}")
        if medium:
            recommendations.append(f"Investigate and apply advanced imputation for medium missing columns: {medium}")
        if low:
            recommendations.append(f"Apply simple imputation strategies for low missing columns: {low}")

        return recommendations

    def _get_duplicate_recommendations(self, full_duplicates: int, feature_duplicates: int) -> List[str]:
        """Generate recommendations for handling duplicates."""
        recommendations = []

        if full_duplicates > 0:
            recommendations.append(f"Remove {full_duplicates} full duplicate records")
        if feature_duplicates > 0:
            recommendations.append(f"Investigate {feature_duplicates} feature duplicates - may indicate data collection issues")

        return recommendations

    def _get_cardinality_recommendations(self, cardinality_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on cardinality analysis."""
        recommendations = []

        high_cardinality_cols = [col for col, info in cardinality_analysis.items()
                               if info['category'] == 'high_cardinality']
        low_cardinality_cols = [col for col, info in cardinality_analysis.items()
                              if info['category'] == 'low_cardinality']

        if high_cardinality_cols:
            recommendations.append(f"Consider feature engineering for high cardinality columns: {high_cardinality_cols}")
        if low_cardinality_cols:
            recommendations.append(f"Consider treating low cardinality columns as categorical: {low_cardinality_cols}")

        return recommendations

    def _get_consistency_recommendations(self, consistency_issues: List[Dict]) -> List[str]:
        """Generate recommendations for consistency issues."""
        recommendations = []

        for issue in consistency_issues:
            col = issue['column']
            if issue['whitespace_issues'] > 0:
                recommendations.append(f"Remove leading/trailing whitespace from '{col}'")
            if issue['case_inconsistencies']:
                recommendations.append(f"Standardize case formatting in '{col}'")

        return recommendations

    def print_summary(self) -> None:
        """Print a concise summary of data quality assessment."""
        print(f"\n{'='*60}")
        print(f"DATA QUALITY SUMMARY - {self.name}")
        print(f"{'='*60}")

        # Generate report for summary
        report = self.generate_quality_report()

        if 'overall_quality' in report:
            quality = report['overall_quality']
            print(f"\nOVERALL QUALITY SCORE: {quality['overall_score']:.1f}/100 (Grade: {quality['grade']})")
            print(f"Assessment: {quality['interpretation']}")

        # Key metrics
        if 'missing_values' in report:
            missing = report['missing_values']['summary']
            print(f"\nMISSING VALUES:")
            print(f"  Total missing: {missing['total_missing_values']:,}")
            print(f"  Overall percentage: {missing['missing_percentage_overall']:.1f}%")
            print(f"  Columns affected: {missing['columns_with_missing']}/{len(self.data.columns)}")

        if 'duplicates' in report:
            duplicates = report['duplicates']
            print(f"\nDUPLICATES:")
            print(f"  Full duplicates: {duplicates['full_duplicates']['count']}")
            print(f"  Feature duplicates: {duplicates['feature_duplicates']['count']}")

        if 'outliers' in report:
            outlier_cols = len(report['outliers'])
            print(f"\nOUTLIERS:")
            print(f"  Analyzed {outlier_cols} numerical columns")

        print(f"\n{'='*60}")


# Test the module
if __name__ == "__main__":
    print("Testing Data Quality Assessment module...")

    # Create sample data with various quality issues
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'price': np.random.normal(100, 20, 100),
        'category': np.random.choice(['A', 'B', 'C', 'A ', ' B', 'c'], 100),  # Inconsistent formatting
        'value': [np.random.normal(50, 10) if i % 4 != 0 else None for i in range(100)],  # Missing values
        'duplicate_col': ['same_value'] * 100,  # Low cardinality
        'unique_id': [f"id_{i}" for i in range(100)]  # High cardinality
    })

    # Add some duplicates
    sample_data.loc[99] = sample_data.loc[0]

    try:
        assessor = DataQualityAssessor(sample_data, "Sample Dataset")
        report = assessor.generate_quality_report()
        assessor.print_summary()
        print("\n✅ Data Quality Assessment module working correctly")
    except Exception as e:
        print(f"\n❌ Error in Data Quality module: {e}")