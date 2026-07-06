import numpy as np


def create_sequences(
    df,
    feature_columns,
    target_column,
    sequence_length=28
):
    """
    Create LSTM sequences for each product-store separately.

    Short groups are padded with the available history so the LSTM can still
    train on the data instead of producing an empty dataset.
    """

    X = []
    y = []

    grouped = df.groupby(["item_id", "store_id"])

    for _, group in grouped:
        group = group.sort_values("date")

        data = group[feature_columns].values
        target = group[target_column].values

        if len(group) == 0:
            continue

        if len(group) < sequence_length:
            padded = np.zeros((sequence_length, data.shape[1]), dtype=float)
            padded[-len(group):] = data
            X.append(padded)
            y.append(target[-1])
            continue

        for i in range(sequence_length, len(group) + 1):
            X.append(data[i - sequence_length:i])
            y.append(target[i - 1])

    if not X:
        return np.empty((0, sequence_length, len(feature_columns))), np.empty((0,))

    return np.array(X), np.array(y)