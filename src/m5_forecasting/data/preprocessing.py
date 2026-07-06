import pandas as pd


def create_sample(sales: pd.DataFrame, n_products: int = 100) -> pd.DataFrame:
    """
    Return the first n_products from the sales dataframe.
    """
    return sales.head(n_products).copy()