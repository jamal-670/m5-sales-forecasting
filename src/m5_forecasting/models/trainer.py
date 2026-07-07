from pathlib import Path

from .evaluator import evaluate_model


def train_and_evaluate_model(model_type, model, X, y, model_path, prediction_fn, save_fn):
    """
    Train a model, save the artifact to disk, generate predictions, and return metrics.
    """
    if model is None:
        raise ValueError("A model instance is required for training.")

    if X is None or y is None:
        raise ValueError("Training features and targets are required.")

    if hasattr(model, "fit"):
        trained_model = model.fit(X, y)
    else:
        raise TypeError(f"Model type {model_type!r} does not expose a fit method.")

    save_path = Path(model_path)
    save_fn(trained_model, save_path)
    predictions = prediction_fn(trained_model, X)
    metrics = evaluate_model(y, predictions)

    return {
        "model_type": model_type,
        "model_path": str(save_path),
        "predictions": predictions,
        "metrics": metrics,
    }
