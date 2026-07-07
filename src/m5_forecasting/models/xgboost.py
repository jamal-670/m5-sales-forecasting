from pathlib import Path

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None


def build_xgboost_model(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, **kwargs):
    """
    Build an XGBoost regressor for sales forecasting.
    """
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is required to use this model. Install it with `pip install xgboost`."
        ) from _XGBOOST_IMPORT_ERROR

    params = {
        "objective": "reg:squarederror",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": random_state,
    }
    params.update(kwargs)

    return XGBRegressor(**params)


def train_xgboost_model(model, X, y):
    """
    Fit an XGBoost regressor on tabular features.
    """
    if X is None or y is None:
        raise ValueError("Features and targets must be provided.")

    feature_frame = X.copy()
    target_series = y.copy()

    if isinstance(feature_frame, pd.DataFrame):
        feature_frame = feature_frame.astype(float)
    if isinstance(target_series, pd.Series):
        target_series = target_series.astype(float)

    model.fit(feature_frame, target_series)
    return model


def predict_xgboost_model(model, X):
    """
    Predict using a trained XGBoost regressor.
    """
    if model is None:
        raise ValueError("A trained model is required for prediction.")

    feature_frame = X.copy()
    if isinstance(feature_frame, pd.DataFrame):
        feature_frame = feature_frame.astype(float)

    predictions = model.predict(feature_frame)
    return np.asarray(predictions)


def save_xgboost_model(model, path):
    """
    Save a trained XGBoost model to disk.
    """
    if model is None:
        raise ValueError("A trained model is required for saving.")

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_path))
    return save_path


def load_xgboost_model(path):
    """
    Load a saved XGBoost model from disk.
    """
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is required to use this model. Install it with `pip install xgboost`."
        ) from _XGBOOST_IMPORT_ERROR

    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    model = build_xgboost_model()
    model.load_model(str(load_path))
    return model


def train_and_save_xgboost_model(model, X, y, path):
    """
    Train an XGBoost model and persist it to disk.
    """
    trained_model = train_xgboost_model(model, X, y)
    save_xgboost_model(trained_model, path)
    return trained_model
