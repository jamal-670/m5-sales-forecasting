"""
Functions for loading the M5 Forecasting dataset.
"""

from pathlib import Path
import pandas as pd


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the M5 Forecasting datasets.
    """

    calendar = pd.read_csv(data_path / "calendar.csv")
    sales = pd.read_csv(data_path / "sales_train_validation.csv")
    prices = pd.read_csv(data_path / "sell_prices.csv")

    return calendar, sales, prices