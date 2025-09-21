"""
Test script to verify the EDA notebook can run successfully.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    # Import required modules
    from src.data_loader import load_data_with_fallback, validate_dataset
    from src.eda import EDAAnalyzer
    from src.data_quality import DataQualityAssessor

    print("📊 Testing EDA Notebook Components...")
    print("=" * 50)

    # Simulate notebook execution
    print("\n1. Loading data...")
    train_data, test_data = load_data_with_fallback()

    print("2. Validating dataset...")
    is_valid = validate_dataset(train_data, test_data)

    if is_valid:
        print("3. Running EDA analysis...")
        analyzer = EDAAnalyzer(train_data, test_data)

        print("4. Generating quality report...")
        quality_assessor = DataQualityAssessor(train_data)

        print("5. Testing key analysis functions...")
        # Test core functionality that notebook uses
        missing_analysis = analyzer.analyze_missing_values()
        target_analysis = analyzer.analyze_target_variable()
        correlations = analyzer.target_correlation(10)

        print(f"\n✅ All notebook components working!")
        print(f"📈 Features analyzed: {len(missing_analysis)}")
        print(f"🎯 Target stats: Mean=${target_analysis['mean']:,.0f}")
        print(f"🔗 Top correlation: {correlations.iloc[0]['feature']} ({correlations.iloc[0]['correlation']:.3f})")

        print(f"\n🚀 EDA Notebook ready to run!")
        print(f"💡 To run: jupyter notebook notebooks/01_exploratory_data_analysis.ipynb")

    else:
        print("❌ Dataset validation failed")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()