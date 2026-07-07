from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def build_lstm_model(input_shape):
    """
    Build an LSTM model for sales forecasting.
    """

    model = Sequential()

    model.add(
        LSTM(
            units=64,
            input_shape=input_shape,
            return_sequences=False
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(32, activation="relu")
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model

def get_callbacks():
    """
    Return callbacks used during training.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    return [early_stopping]


def save_lstm_model(model, path):
    """
    Save a trained LSTM model to disk.
    """
    if model is None:
        raise ValueError("A trained model is required for saving.")

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    return save_path


def load_lstm_model(path):
    """
    Load a saved LSTM model from disk.
    """
    from tensorflow.keras.models import load_model

    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    return load_model(str(load_path))
