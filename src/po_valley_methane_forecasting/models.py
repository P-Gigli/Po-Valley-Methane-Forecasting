from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_candidate_models() -> dict[str, BaseEstimator]:
    """
    Return the unfitted candidate regression models.
    """
    return {
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Random Forest — default": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest — regularized": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=25,
            max_features="sqrt",
            max_samples=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }
