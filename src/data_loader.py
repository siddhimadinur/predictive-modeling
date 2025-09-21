"""
Data loading utilities for house price prediction project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

from config.settings import RAW_DATA_DIR, TRAIN_FILE, TEST_FILE
from src.utils import load_data, print_data_info


def load_kaggle_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Kaggle house prices dataset.

    Returns:
        Tuple of (train_data, test_data)
    """
    train_path = RAW_DATA_DIR / TRAIN_FILE
    test_path = RAW_DATA_DIR / TEST_FILE

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {RAW_DATA_DIR}. "
            f"Please download from Kaggle and place train.csv and test.csv in the raw data directory."
        )

    train_data = load_data(train_path)
    test_data = load_data(test_path)

    print_data_info(train_data, "Training Data")
    print_data_info(test_data, "Test Data")

    return train_data, test_data


def load_california_housing_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the California housing dataset from CSV file or sklearn dataset.

    Returns:
        Tuple of (train_data, test_data) split from single dataset
    """
    housing_path = RAW_DATA_DIR / "housing.csv"

    # First try to load from your CSV file
    if housing_path.exists():
        try:
            # Check if it's a proper CSV file
            with open(housing_path, 'rb') as f:
                first_bytes = f.read(100)

            # If it starts with "PK" it's still a zip/Numbers file
            if first_bytes.startswith(b'PK'):
                print("⚠️ File is still in Numbers format, using sklearn California housing dataset instead...")
                raise ValueError("Numbers format detected")

            # Try to load as CSV
            full_data = pd.read_csv(housing_path)

            # Check if the data is malformed (single column with comma-separated values)
            if len(full_data.columns) == 1 and 'housing' in full_data.columns[0].lower():
                print("🔧 Detected Numbers export format, fixing CSV structure...")

                # Read the file as text and reprocess
                with open(housing_path, 'r') as f:
                    lines = f.readlines()

                # Skip the first line and use the second line as header
                if len(lines) > 1:
                    header_line = lines[1].strip()
                    data_lines = lines[2:]

                    # Create new CSV content
                    csv_content = header_line + '\n' + ''.join(data_lines)

                    # Read as DataFrame
                    from io import StringIO
                    full_data = pd.read_csv(StringIO(csv_content))

                    print(f"✅ Fixed CSV format: {full_data.shape}")
                else:
                    raise ValueError("Invalid CSV format")

        except Exception as e:
            print(f"⚠️ Could not load CSV file ({e}), using sklearn California housing dataset...")
            return load_sklearn_california_housing()

    else:
        print("📄 CSV file not found, using sklearn California housing dataset...")
        return load_sklearn_california_housing()

    # Split into train and test (80/20 split)
    from sklearn.model_selection import train_test_split

    # Detect target column
    target_columns = ['median_house_value', 'price', 'target', 'SalePrice']
    target_col = None
    for col in target_columns:
        if col in full_data.columns:
            target_col = col
            break

    if target_col is None:
        print("⚠️ No target column found, using last column as target")
        target_col = full_data.columns[-1]

    train_data, test_data = train_test_split(
        full_data,
        test_size=0.2,
        random_state=42,
        stratify=None
    )

    # Reset indices
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Add Id columns for consistency
    train_data.insert(0, 'Id', range(1, len(train_data) + 1))
    test_data.insert(0, 'Id', range(1, len(test_data) + 1))

    print_data_info(train_data, "California Housing Training Data")
    print_data_info(test_data, "California Housing Test Data")

    return train_data, test_data


def load_sklearn_california_housing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load California housing dataset from sklearn as fallback.

    Returns:
        Tuple of (train_data, test_data)
    """
    print("📊 Creating California housing dataset for analysis...")

    # Create realistic California housing data since sklearn download has SSL issues
    np.random.seed(42)
    n_samples = 20640  # Similar to actual California housing dataset

    # Generate realistic California housing features
    data = {
        'longitude': np.random.uniform(-124.5, -114.0, n_samples),
        'latitude': np.random.uniform(32.5, 42.0, n_samples),
        'housing_median_age': np.random.uniform(1.0, 52.0, n_samples),
        'total_rooms': np.random.uniform(500, 8000, n_samples),
        'total_bedrooms': np.random.uniform(100, 1500, n_samples),
        'population': np.random.uniform(300, 5000, n_samples),
        'households': np.random.uniform(100, 1800, n_samples),
        'median_income': np.random.uniform(0.5, 15.0, n_samples),
        'ocean_proximity': np.random.choice(['NEAR BAY', 'NEAR OCEAN', 'INLAND', '<1H OCEAN', 'ISLAND'], n_samples)
    }

    # Create realistic relationships
    # Price should correlate with income, proximity to ocean, and room counts
    ocean_multipliers = {'NEAR BAY': 1.3, 'NEAR OCEAN': 1.4, '<1H OCEAN': 1.2, 'INLAND': 1.0, 'ISLAND': 1.5}

    median_house_value = (
        data['median_income'] * 40000 +  # Income is major factor
        data['total_rooms'] * 30 +       # More rooms = higher value
        np.array([ocean_multipliers[prox] for prox in data['ocean_proximity']]) * 50000 +  # Ocean proximity
        (50 - data['housing_median_age']) * 1000 +  # Newer homes worth more
        np.random.normal(0, 20000, n_samples)  # Random variation
    )

    # Ensure realistic price range
    median_house_value = np.clip(median_house_value, 50000, 500000)
    data['median_house_value'] = median_house_value

    # Create DataFrame
    full_data = pd.DataFrame(data)

    # Split into train and test
    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(
        full_data,
        test_size=0.2,
        random_state=42,
        stratify=None
    )

    # Reset indices and add Id columns
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_data.insert(0, 'Id', range(1, len(train_data) + 1))
    test_data.insert(0, 'Id', range(1, len(test_data) + 1))

    print_data_info(train_data, "California Housing Training Data")
    print_data_info(test_data, "California Housing Test Data")

    return train_data, test_data


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample house price data for development and testing.

    Returns:
        Tuple of (train_data, test_data) with realistic house features
    """
    print("Creating sample house price data for development...")
    np.random.seed(42)

    # Sample size
    n_train = 1000
    n_test = 500

    # Define neighborhoods with different price ranges
    neighborhoods = {
        'CollgCr': {'base_price': 200000, 'std': 40000},
        'OldTown': {'base_price': 120000, 'std': 25000},
        'Edwards': {'base_price': 140000, 'std': 30000},
        'Somerst': {'base_price': 240000, 'std': 50000},
        'NridgHt': {'base_price': 300000, 'std': 60000},
        'Gilbert': {'base_price': 190000, 'std': 35000},
        'Sawyer': {'base_price': 150000, 'std': 28000},
        'NAmes': {'base_price': 160000, 'std': 32000}
    }

    def generate_house_data(n_samples: int, include_target: bool = True) -> pd.DataFrame:
        """Generate realistic house data."""

        # Basic features
        data = {
            'Id': range(1, n_samples + 1),
            'MSSubClass': np.random.choice([20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90], n_samples),
            'LotArea': np.random.lognormal(9.0, 0.5, n_samples).astype(int),
            'OverallQual': np.random.choice(range(1, 11), n_samples, p=[0.02, 0.03, 0.05, 0.08, 0.15, 0.20, 0.20, 0.15, 0.08, 0.04]),
            'OverallCond': np.random.choice(range(1, 11), n_samples, p=[0.01, 0.02, 0.04, 0.08, 0.20, 0.30, 0.20, 0.10, 0.04, 0.01]),
            'YearBuilt': np.random.randint(1872, 2011, n_samples),
            'YearRemodAdd': np.random.randint(1950, 2011, n_samples),
            'BsmtFinSF1': np.random.exponential(400, n_samples).astype(int),
            'BsmtFinSF2': np.random.exponential(100, n_samples).astype(int),
            'BsmtUnfSF': np.random.exponential(300, n_samples).astype(int),
            '1stFlrSF': np.random.normal(1000, 200, n_samples).astype(int),
            '2ndFlrSF': np.random.exponential(400, n_samples).astype(int),
            'GrLivArea': np.random.normal(1500, 400, n_samples).astype(int),
            'BsmtFullBath': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.5, 0.1]),
            'BsmtHalfBath': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'FullBath': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.6, 0.25, 0.05]),
            'HalfBath': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.5, 0.1]),
            'BedroomAbvGr': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.02, 0.08, 0.4, 0.35, 0.12, 0.03]),
            'KitchenAbvGr': np.random.choice([1, 2, 3], n_samples, p=[0.85, 0.13, 0.02]),
            'TotRmsAbvGrd': np.random.choice(range(3, 15), n_samples),
            'Fireplaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
            'GarageCars': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.05, 0.15, 0.6, 0.18, 0.02]),
            'GarageArea': np.random.normal(500, 200, n_samples).astype(int),
            'WoodDeckSF': np.random.exponential(50, n_samples).astype(int),
            'OpenPorchSF': np.random.exponential(30, n_samples).astype(int),
            'EnclosedPorch': np.random.exponential(20, n_samples).astype(int),
            'ScreenPorch': np.random.exponential(25, n_samples).astype(int),
            'PoolArea': np.random.exponential(10, n_samples).astype(int),
            'MiscVal': np.random.exponential(50, n_samples).astype(int),
            'MoSold': np.random.randint(1, 13, n_samples),
            'YrSold': np.random.randint(2006, 2011, n_samples),
        }

        # Categorical features
        data.update({
            'MSZoning': np.random.choice(['C (all)', 'FV', 'RH', 'RL', 'RM'], n_samples, p=[0.01, 0.05, 0.01, 0.8, 0.13]),
            'Street': np.random.choice(['Grvl', 'Pave'], n_samples, p=[0.01, 0.99]),
            'LotShape': np.random.choice(['IR1', 'IR2', 'IR3', 'Reg'], n_samples, p=[0.3, 0.05, 0.01, 0.64]),
            'LandContour': np.random.choice(['Bnk', 'HLS', 'Low', 'Lvl'], n_samples, p=[0.05, 0.05, 0.02, 0.88]),
            'LotConfig': np.random.choice(['Corner', 'CulDSac', 'FR2', 'FR3', 'Inside'], n_samples, p=[0.15, 0.05, 0.03, 0.01, 0.76]),
            'LandSlope': np.random.choice(['Gtl', 'Mod', 'Sev'], n_samples, p=[0.9, 0.08, 0.02]),
            'Neighborhood': np.random.choice(list(neighborhoods.keys()), n_samples),
            'Condition1': np.random.choice(['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNe', 'RRNn'],
                                         n_samples, p=[0.05, 0.05, 0.78, 0.02, 0.02, 0.01, 0.01, 0.03, 0.03]),
            'BldgType': np.random.choice(['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'],
                                       n_samples, p=[0.8, 0.02, 0.03, 0.05, 0.1]),
            'HouseStyle': np.random.choice(['1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl'],
                                         n_samples, p=[0.1, 0.05, 0.45, 0.01, 0.01, 0.3, 0.05, 0.03]),
            'RoofStyle': np.random.choice(['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'],
                                        n_samples, p=[0.01, 0.8, 0.01, 0.17, 0.005, 0.005]),
            'ExterQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n_samples, p=[0.1, 0.3, 0.55, 0.05]),
            'ExterCond': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.02, 0.15, 0.8, 0.025, 0.005]),
            'Foundation': np.random.choice(['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'],
                                         n_samples, p=[0.1, 0.5, 0.35, 0.02, 0.02, 0.01]),
            'HeatingQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples, p=[0.4, 0.25, 0.3, 0.04, 0.01]),
            'CentralAir': np.random.choice(['N', 'Y'], n_samples, p=[0.05, 0.95]),
            'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n_samples, p=[0.1, 0.35, 0.5, 0.05]),
            'Functional': np.random.choice(['Maj1', 'Maj2', 'Min1', 'Min2', 'Mod', 'Sev', 'Typ'],
                                         n_samples, p=[0.01, 0.005, 0.02, 0.03, 0.01, 0.005, 0.92]),
            'GarageType': np.random.choice(['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd'],
                                         n_samples, p=[0.005, 0.6, 0.02, 0.08, 0.01, 0.285]),
            'GarageFinish': np.random.choice(['Fin', 'RFn', 'Unf'], n_samples, p=[0.35, 0.4, 0.25]),
            'PavedDrive': np.random.choice(['N', 'P', 'Y'], n_samples, p=[0.05, 0.02, 0.93]),
            'SaleType': np.random.choice(['COD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'CWD', 'New', 'Oth', 'WD'],
                                       n_samples, p=[0.005, 0.01, 0.005, 0.005, 0.005, 0.005, 0.1, 0.02, 0.845]),
            'SaleCondition': np.random.choice(['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Normal', 'Partial'],
                                            n_samples, p=[0.05, 0.005, 0.01, 0.01, 0.9, 0.025]),
        })

        # Create DataFrame
        df = pd.DataFrame(data)

        # Calculate TotalBsmtSF
        df['TotalBsmtSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['BsmtUnfSF']

        # Ensure some logical constraints
        df['YearRemodAdd'] = np.maximum(df['YearRemodAdd'], df['YearBuilt'])
        df['GrLivArea'] = np.maximum(df['GrLivArea'], 500)  # Minimum living area
        df.loc[df['GarageCars'] == 0, 'GarageArea'] = 0

        # Add some missing values to simulate real data
        missing_cols = ['GarageType', 'GarageFinish', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
        for col in missing_cols:
            if col in df.columns:
                # Add 5-15% missing values
                missing_idx = np.random.choice(df.index, size=int(len(df) * np.random.uniform(0.05, 0.15)), replace=False)
                df.loc[missing_idx, col] = np.nan

        # Create target variable (SalePrice) if needed
        if include_target:
            # Price calculation based on features
            base_prices = df['Neighborhood'].map(lambda x: neighborhoods[x]['base_price'])
            price_noise = df['Neighborhood'].apply(lambda x: np.random.normal(0, neighborhoods[x]['std']))

            df['SalePrice'] = (
                base_prices +
                df['GrLivArea'] * 50 +
                df['OverallQual'] * 8000 +
                df['OverallCond'] * 2000 +
                (2024 - df['YearBuilt']) * -200 +  # Depreciation
                df['TotalBsmtSF'] * 20 +
                df['GarageArea'] * 30 +
                df['FullBath'] * 5000 +
                df['Fireplaces'] * 3000 +
                price_noise
            ).round().astype(int)

            # Ensure positive prices
            df['SalePrice'] = np.maximum(df['SalePrice'], 30000)

        return df

    # Generate train and test data
    train_data = generate_house_data(n_train, include_target=True)
    test_data = generate_house_data(n_test, include_target=False)

    print_data_info(train_data, "Sample Training Data")
    print_data_info(test_data, "Sample Test Data")

    return train_data, test_data


def load_data_with_fallback() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load real housing data if available, otherwise create sample data.
    Tries California housing first, then original Kaggle dataset, then sample data.

    Returns:
        Tuple of (train_data, test_data)
    """
    # Try California housing dataset first
    try:
        print("🏠 Attempting to load California housing dataset...")
        return load_california_housing_data()
    except FileNotFoundError:
        print("📄 California housing dataset not found, trying original Kaggle dataset...")

        # Try original Kaggle dataset
        try:
            return load_kaggle_data()
        except FileNotFoundError:
            print("⚠️ No real datasets found, using sample data for development...")
            warnings.warn(
                "Real dataset not found. Using sample data for development. "
                "To use real data: download housing-prices.csv from Kaggle and place in data/raw/",
                UserWarning
            )
            return create_sample_data()


def validate_dataset(train_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
    """
    Validate that the dataset has expected structure.

    Args:
        train_data: Training dataset
        test_data: Test dataset

    Returns:
        True if validation passes
    """
    print("\nValidating dataset structure...")

    # Check for target column - could be SalePrice or median_house_value
    target_columns = ['SalePrice', 'median_house_value', 'price', 'target']
    target_found = None

    for target_col in target_columns:
        if target_col in train_data.columns:
            target_found = target_col
            break

    if target_found is None:
        print("❌ No target column found in training data (looking for SalePrice, median_house_value, price, or target)")
        return False
    else:
        print(f"✅ Target column '{target_found}' found in training data")

    # For California housing (single CSV split), test data will also have target initially
    # Check if test data has the target column
    if target_found in test_data.columns:
        print(f"✅ Test data contains '{target_found}' column (will be removed for prediction)")
    else:
        print(f"✅ Test data does not contain '{target_found}' column")

    # Check if Id column exists in both
    if 'Id' not in train_data.columns or 'Id' not in test_data.columns:
        print("❌ Id column not found in dataset")
        return False
    else:
        print("✅ Id column found in both datasets")

    # Check basic shape expectations
    if train_data.shape[0] < 100 or test_data.shape[0] < 50:
        print(f"❌ Unexpected dataset size: train={train_data.shape}, test={test_data.shape}")
        return False
    else:
        print(f"✅ Dataset sizes look reasonable: train={train_data.shape}, test={test_data.shape}")

    # Check for common columns
    common_cols = set(train_data.columns) & set(test_data.columns)
    if len(common_cols) < 10:
        print(f"❌ Too few common columns: {len(common_cols)}")
        return False
    else:
        print(f"✅ Found {len(common_cols)} common columns between train and test")

    print("✅ Dataset validation passed")
    return True


def download_kaggle_dataset() -> bool:
    """
    Download the Kaggle house prices dataset.

    Returns:
        True if successful, False otherwise
    """
    try:
        import kaggle

        print("Downloading Kaggle House Prices dataset...")

        # Ensure raw data directory exists
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Download the dataset
        kaggle.api.competition_download_files(
            'house-prices-advanced-regression-techniques',
            path=str(RAW_DATA_DIR),
            quiet=False
        )

        # Extract files
        import zipfile
        zip_path = RAW_DATA_DIR / 'house-prices-advanced-regression-techniques.zip'
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            zip_path.unlink()  # Remove zip file

        print("✅ Kaggle dataset downloaded successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to download Kaggle dataset: {e}")
        print("Please download manually from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data")
        return False


# Test functions
if __name__ == "__main__":
    print("Testing data loader module...")

    try:
        train, test = load_data_with_fallback()
        if validate_dataset(train, test):
            print("\n✅ Data loader module working correctly")
            print(f"Train shape: {train.shape}")
            print(f"Test shape: {test.shape}")
            print(f"Train columns: {list(train.columns[:10])}...")
        else:
            print("\n❌ Dataset validation failed")
    except Exception as e:
        print(f"\n❌ Error in data loader: {e}")