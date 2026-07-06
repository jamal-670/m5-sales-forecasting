from sklearn.preprocessing import MinMaxScaler


def scale_features(df):
    """
    Scale numerical features using MinMaxScaler.
    """

    df = df.copy()

    numerical_columns = [
        "sell_price",
        "lag_1",
        "lag_7",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend"
    ]

    scaler = MinMaxScaler()

    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, scaler