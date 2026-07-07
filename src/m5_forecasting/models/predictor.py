import numpy as np


def predict_with_model(model, X):
    """
    Generic prediction wrapper for models that expose a predict method.
    """
    if model is None:
        raise ValueError("A trained model is required for prediction.")

    predictions = model.predict(X)
    return np.asarray(predictions)
