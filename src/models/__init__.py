"""
Machine learning models for California housing price prediction.
"""

from .base_model import BaseModel
from .linear_models import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetModel
)
from .ensemble_models import (
    RandomForestModel,
    GradientBoostingModel,
    ExtraTreesModel,
    XGBOOST_AVAILABLE
)

__all__ = [
    'BaseModel',
    'LinearRegressionModel',
    'RidgeRegressionModel',
    'LassoRegressionModel',
    'ElasticNetModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'ExtraTreesModel',
]

if XGBOOST_AVAILABLE:
    from .ensemble_models import XGBoostModel
    __all__.append('XGBoostModel')
