import numpy as np
import pandas as pd

from m5_forecasting.data.sequence import create_sequences


def test_create_sequences_pads_short_groups_to_sequence_length():
    df = pd.DataFrame(
        {
            "item_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "store_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "sales": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    X, y = create_sequences(df, ["feature"], "sales", sequence_length=5)

    assert X.ndim == 3
    assert X.shape[1:] == (5, 1)
    assert y.shape == (2,)
    assert np.isfinite(X).all()
