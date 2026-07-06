def train_val_test_split(X, y, train_size=0.7, val_size=0.15):
    """
    Split sequences into train, validation and test sets.
    """

    n = len(X)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )
