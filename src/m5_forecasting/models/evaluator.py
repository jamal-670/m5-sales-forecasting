import numpy as np


def evaluate_model(y_true, y_pred):
    """
    Compute common regression metrics for a model prediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
