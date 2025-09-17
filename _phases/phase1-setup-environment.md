# Phase 1: Project Setup & Environment

## Overview
Set up the complete project structure, development environment, and essential dependencies for the house price prediction project.

## Objectives
- Create organized project directory structure
- Set up Python virtual environment
- Install required dependencies
- Configure development tools
- Initialize version control
- Set up basic configuration files

## Step-by-Step Implementation

### 1.1 Create Project Directory Structure
```bash
# Create main directories
mkdir -p data/{raw,processed}
mkdir -p notebooks
mkdir -p src
mkdir -p models/trained_models
mkdir -p app
mkdir -p tests
mkdir -p config
```

**Test**: Verify directory structure exists
```bash
tree . -L 2
```

### 1.2 Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Verify Python version
python --version
```

**Test**: Confirm virtual environment is active
```bash
which python
pip list  # Should show minimal packages
```

### 1.3 Create Requirements File
Create `requirements.txt` with essential packages:
```txt
# Data Processing
pandas>=1.5.0
numpy>=1.21.0

# Machine Learning
scikit-learn>=1.1.0
xgboost>=1.6.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Web Application
streamlit>=1.15.0

# Development Tools
jupyter>=1.0.0
ipykernel>=6.15.0

# Data Handling
requests>=2.28.0

# Model Persistence
joblib>=1.2.0
pickle-mixin>=1.0.2

# Utilities
python-dotenv>=0.19.0
```

**Test**: Install dependencies and verify
```bash
pip install -r requirements.txt
pip list | grep -E "(pandas|scikit-learn|streamlit)"
```

### 1.4 Create Configuration Files

#### 1.4.1 Create `.gitignore`
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/raw/*.csv
data/processed/*.csv
*.pkl
*.joblib

# Model files
models/trained_models/*.pkl
models/trained_models/*.joblib

# Environment variables
.env

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/
```

#### 1.4.2 Create `config/settings.py`
```python
"""
Configuration settings for the house price prediction project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data settings
DATASET_URL = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering
NUMERICAL_FEATURES = [
    'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
    'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]

CATEGORICAL_FEATURES = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
    'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
    'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
    'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

# Model hyperparameters
MODEL_PARAMS = {
    'linear_regression': {},
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}

# Streamlit settings
STREAMLIT_CONFIG = {
    'page_title': 'House Price Predictor',
    'page_icon': '🏠',
    'layout': 'wide'
}
```

**Test**: Verify configuration loads correctly
```python
python -c "from config.settings import PROJECT_ROOT; print(f'Project root: {PROJECT_ROOT}')"
```

### 1.5 Create Initial Python Modules

#### 1.5.1 Create `src/__init__.py`
```python
"""
Source code for house price prediction project.
"""
```

#### 1.5.2 Create `src/utils.py`
```python
"""
Utility functions for the house price prediction project.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def ensure_dir_exists(path: Path) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def save_data(data: pd.DataFrame, filepath: Path) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        data: DataFrame to save
        filepath: Path to save file
    """
    ensure_dir_exists(filepath.parent)
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        Loaded DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = pd.read_csv(filepath)
    print(f"Data loaded from {filepath} - Shape: {data.shape}")
    return data


def print_data_info(data: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic information about a DataFrame.

    Args:
        data: DataFrame to analyze
        name: Name for the dataset
    """
    print(f"\n{name} Information:")
    print(f"Shape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
```

**Test**: Verify utility functions work
```python
python -c "from src.utils import setup_logging; logger = setup_logging(); logger.info('Utils module working correctly')"
```

### 1.6 Initialize Git Repository
```bash
# Initialize git repository
git init

# Add initial files
git add .

# Create initial commit
git commit -m "Initial project setup with directory structure and configuration"
```

**Test**: Verify git repository initialized
```bash
git status
git log --oneline
```

### 1.7 Create Development Helper Scripts

#### 1.7.1 Create `scripts/setup.sh`
```bash
#!/bin/bash
# Development environment setup script

echo "Setting up House Price Prediction project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p models/trained_models
mkdir -p notebooks
mkdir -p tests

echo "Setup complete! Activate your environment with: source venv/bin/activate"
```

Make script executable:
```bash
chmod +x scripts/setup.sh
mkdir -p scripts
```

#### 1.7.2 Create `scripts/run_tests.sh`
```bash
#!/bin/bash
# Test runner script

echo "Running project tests..."

# Activate virtual environment
source venv/bin/activate

# Run Python syntax checks
echo "Checking Python syntax..."
python -m py_compile src/*.py
python -m py_compile config/*.py

# Test imports
echo "Testing module imports..."
python -c "import src.utils; print('✓ Utils module imports successfully')"
python -c "from config.settings import PROJECT_ROOT; print('✓ Settings module imports successfully')"

# Test directory structure
echo "Verifying directory structure..."
python -c "
import os
from pathlib import Path

required_dirs = ['data/raw', 'data/processed', 'src', 'models/trained_models', 'app', 'notebooks']
missing_dirs = [d for d in required_dirs if not Path(d).exists()]

if missing_dirs:
    print(f'✗ Missing directories: {missing_dirs}')
    exit(1)
else:
    print('✓ All required directories exist')
"

echo "All tests passed!"
```

Make script executable:
```bash
chmod +x scripts/run_tests.sh
```

### 1.8 Final Verification

**Test**: Run complete environment test
```bash
# Run the test script
./scripts/run_tests.sh

# Verify Python environment
python -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import streamlit as st
print('✓ All major packages imported successfully')
print(f'✓ Pandas version: {pd.__version__}')
print(f'✓ Scikit-learn version: {sklearn.__version__}')
print(f'✓ XGBoost version: {xgb.__version__}')
print(f'✓ Streamlit version: {st.__version__}')
"
```

## Deliverables
- [ ] Complete project directory structure
- [ ] Python virtual environment with all dependencies
- [ ] Configuration files (settings.py, .gitignore)
- [ ] Requirements.txt with pinned versions
- [ ] Initial utility modules
- [ ] Git repository initialization
- [ ] Development helper scripts
- [ ] Comprehensive test suite for setup verification

## Success Criteria
- All directories created as per project structure
- Virtual environment activated and dependencies installed
- Configuration files load without errors
- Git repository initialized with initial commit
- Test scripts run successfully
- All major packages import correctly

## Next Phase
Proceed to **Phase 2: Data Acquisition & Exploration** once all setup verification tests pass.