"""
Functions for reshaping and merging M5 datasets.
"""

import pandas as pd


def melt_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the sales dataset from wide format to long format.

    Parameters
    ----------
    sales : pd.DataFrame
        Sales dataframe in wide format.

    Returns
    -------
    pd.DataFrame
        Sales dataframe in long format.
    """

    sales_long = pd.melt(
        frame=sales,
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ],
        value_vars=sales.columns[6:],
        var_name="d",
        value_name="sales",
    )

    return sales_long

import pandas as pd


def merge_calendar(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge melted sales data with calendar data.
    """

    merged = sales_long.merge(
        calendar,
        on="d",
        how="left"
    )

    return merged

def merge_prices(
    sales_calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge sales data with sell prices.
    """

    return sales_calendar.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )