import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the date column.
    """

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def create_lag_features(df):
    """
    Create lag features for each product-store combination.
    """

    df = df.copy()

    df = df.sort_values(
        by=["item_id", "store_id", "date"]
    )

    group = df.groupby(
        ["item_id", "store_id"]
    )

    df["lag_1"] = group["sales"].shift(1)
    df["lag_7"] = group["sales"].shift(7)
    df["lag_28"] = group["sales"].shift(28)

    return df

def create_rolling_features(df):
    """
    Create rolling mean features.
    """

    df = df.copy()

    df = df.sort_values(
        by=["item_id", "store_id", "date"]
    )

    group = df.groupby(
        ["item_id", "store_id"]
    )

    df["rolling_mean_7"] = (
        group["sales"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    df["rolling_mean_28"] = (
        group["sales"]
        .transform(lambda x: x.shift(1).rolling(28).mean())
    )

    return df