# California Housing Price Predictor

A machine learning system that predicts California house prices using ensemble models, paired with an interactive Streamlit dashboard for real-time price estimation.

**Champion Model**: Gradient Boosting — R² 0.83 | RMSE $47,296

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)

---

## Features

- **8 trained regression models** compared and evaluated (Linear, Ridge, Lasso, ElasticNet, Polynomial, Random Forest, Gradient Boosting, Extra Trees)
- **Automated data pipeline** with cleaning, imputation, and feature engineering (geographic, density, income-based, polynomial features)
- **Interactive web app** with real-time predictions, model insights, and California region presets
- **Comprehensive evaluation** with cross-validation, multiple metrics, and hyperparameter tuning

## Quick Start

```bash
# Clone the repository
git clone https://github.com/siddhimadinur/predictive-modeling.git
cd predictive-modeling

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Run the Web App

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### Train Models from Scratch

```bash
python scripts/train_models.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
├── app/                        # Streamlit web application
│   ├── streamlit_app.py        # Main dashboard UI
│   ├── config.py               # App configuration & region presets
│   ├── components/             # UI input form components
│   └── utils/                  # Model loading utilities
├── src/                        # Core ML pipeline
│   ├── data_loader.py          # Dataset loading (sklearn/CSV)
│   ├── data_pipeline.py        # End-to-end preprocessing
│   ├── feature_engineering.py  # Geographic, density & income features
│   ├── model_training.py       # Training, tuning & evaluation
│   └── models/                 # Model implementations
│       ├── base_model.py       # Abstract base class
│       ├── linear_models.py    # Linear/Ridge/Lasso/ElasticNet/Polynomial
│       └── ensemble_models.py  # Random Forest/Gradient Boosting/XGBoost
├── models/trained_models/      # Serialized trained models (.pkl)
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_processing_feature_engineering.ipynb
│   └── 03_model_development_training.ipynb
├── scripts/                    # Utility & training scripts
├── tests/                      # Unit, model & integration tests
├── config/                     # Project settings
└── _phases/                    # Phase-by-phase documentation
```

## Dataset

**Source**: [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (sklearn built-in) — 20,640 samples

| Feature | Description |
|---|---|
| `median_income` | Median household income (in $10K) |
| `housing_median_age` | Median age of houses in the block group |
| `ave_rooms` | Average rooms per household |
| `ave_bedrooms` | Average bedrooms per household |
| `population` | Block group population |
| `ave_occupancy` | Average household size |
| `latitude` | Geographic latitude |
| `longitude` | Geographic longitude |

**Target**: `median_house_value` — Median house value in dollars

## Model Performance

| Model | RMSE | R² |
|---|---|---|
| **Gradient Boosting** | $13,656 | 0.978 |
| Extra Trees | $14,304 | 0.976 |
| Polynomial Regression | $14,348 | 0.976 |
| Random Forest | $14,777 | 0.974 |
| Lasso | $18,010 | 0.962 |
| Linear Regression | $18,028 | 0.961 |
| Ridge | $18,044 | 0.961 |
| ElasticNet | $30,085 | 0.893 |

*Training metrics on 16,512 samples. Validation R² for champion model: 0.829 on 4,128 held-out samples.*

## Web App

The Streamlit dashboard provides:

- **Price Prediction** — Input property features or select a California region preset (SF Bay, LA, San Diego, Sacramento, Central Valley) and get instant price estimates
- **Model Insights** — Compare model performance metrics and view feature importance rankings
- **California Explorer** — Interactive geographic and demographic analysis

## Tech Stack

| Category | Libraries |
|---|---|
| ML & Data | scikit-learn, XGBoost, pandas, NumPy, SciPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Notebooks | Jupyter, IPython |

## License

MIT
