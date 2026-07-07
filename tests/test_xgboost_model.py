import numpy as np
import pandas as pd

from m5_forecasting.models.evaluator import evaluate_model
from m5_forecasting.models.xgboost import (
    build_xgboost_model,
    train_xgboost_model,
    predict_xgboost_model,
    save_xgboost_model,
    load_xgboost_model,
)


def test_xgboost_model_can_fit_and_predict():
    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    y = pd.Series([3.0, 6.0, 9.0, 12.0, 15.0, 18.0])

    model = build_xgboost_model(n_estimators=10, max_depth=2, random_state=42)
    trained_model = train_xgboost_model(model, X, y)
    predictions = predict_xgboost_model(trained_model, X)

    assert trained_model is not None
    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()
    assert predictions.shape[0] == X.shape[0]


def test_xgboost_model_can_be_saved_and_evaluated(tmp_path):
    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    y = pd.Series([3.0, 6.0, 9.0, 12.0, 15.0, 18.0])

    model = build_xgboost_model(n_estimators=10, max_depth=2, random_state=42)
    trained_model = train_xgboost_model(model, X, y)
    path = tmp_path / "xgboost-model.json"

    save_xgboost_model(trained_model, path)
    loaded_model = load_xgboost_model(path)
    predictions = predict_xgboost_model(loaded_model, X)
    metrics = evaluate_model(y, predictions)

    assert path.exists()
    assert len(predictions) == len(y)
    assert set(metrics.keys()) >= {"mae", "rmse", "mape"}
