from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(df):
    """
    Label encode categorical columns.
    """

    df = df.copy()

    categorical_columns = [
        "item_id",
        "store_id",
        "dept_id",
        "cat_id",
        "state_id"
    ]

    encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    return df, encoders